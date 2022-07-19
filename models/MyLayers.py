from importlib.metadata import distribution
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from utils import extract_dis_from_attention


def create_window_mask(l_seg, window_size):
    """
    :param l_seg: query segment length
    :param window_size: an odd, represents window size
    :return: a mask matrix with shape of (l_seg, l_seg + 2 * b//2), the positions of elements which participate in calculation are 1
    """
    mask = torch.ones(l_seg, l_seg + 2 * (window_size // 2))
    mask = torch.tril(mask, diagonal=window_size-1)
    mask = torch.triu(mask, diagonal=0)
    return mask.bool()


def scalar_dot_attn(queries, keys, values, mask=None, rpe=None, return_attn=False):
    """
    :param queries: (B, H, D_q, l_q)
    :param keys:    (B, H, D_k, l_k)
    :param values:  (B, H, D_v, l_k)
    :param mask:    (B, l_q, l_k)
    :param rpe:     (1, H, l_q, l_k)
    :return:        out: (B, D_out, l_q)  scores: (B, l, window_size)
    """
    B, H, D_q, L_q = queries.shape
    _, _, D_k, L_k = keys.shape
    _, _, D_v, L_v = values.shape
    assert L_k == L_v
    assert D_q == D_k

    scores = torch.einsum("bhdl,bhdn->bhln", queries, keys) / math.sqrt(D_q)  # (B, H, l_q, l_k)

    if rpe is not None:
        scores = scores + rpe  # (B, H, l_q, l_k) + (1, H, l_q, l_k) = (B, H, l_q, l_k)

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1), -1e9)  # Fill elements of the tensor with -inf where mask is True
    
    scores_norm = F.softmax(scores, dim=-1)  # (B, H, l_q, l_k)
    V = torch.einsum("bhdn,bhln->bhdl", values, scores_norm)  # (B, H, D, l_q)
    V = V.reshape(B, -1, L_q)  # (B, D_out, l_q)

    if return_attn:
        attn = scores_norm
    else:
        attn = None

    return V, attn
    

class LearnableAPE(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(LearnableAPE, self).__init__()
        self.embeddings = nn.Embedding(max_len, d_model)
        self.register_buffer('position_ids', torch.arange(max_len))

    def forward(self, x):
        """
        x: (B, D, L)
        return: (B, D, L)
        """
        position_ids = self.position_ids[:x.size(-1)]
        ape = self.embeddings(position_ids).transpose(0, 1)
        return x + ape[None, :, :]


class SinusoidalAPE(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(SinusoidalAPE, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, L, D)
        self.register_buffer('ape', pe)

    def forward(self, x):  
        """
        x: (B, D, L)
        return: (B, D, L)
        """
        x = x + self.ape[:, :x.size(2), :].transpose(1, 2)
        return x


class RPE(nn.Module):
    def __init__(self, window_size, h):
        super(RPE, self).__init__()
        self.h = h
        self.window_size = window_size
        self.embeddings = nn.Embedding(window_size, h)

    def compute_bias(self, l_q, l_k):
        """
        l_q: length of the query
        l_k: length of the key
        return: [1 h l_q l_k]
        """
        query_position = torch.arange(l_q, dtype=torch.long, device=self.embeddings.weight.device)[:, None]
        key_position = torch.arange(l_k, dtype=torch.long, device=self.embeddings.weight.device)[None, :]

        relative_position = query_position - key_position
        if l_q == l_k:
            rpe = torch.clamp(relative_position, -(self.window_size // 2), self.window_size // 2) + self.window_size // 2
        elif l_q + 2 * (self.window_size // 2) == l_k:
            rpe = torch.clamp(relative_position, -2 * (self.window_size // 2), 0) + 2 * (self.window_size // 2)
        else:
            raise RuntimeError("Wrong RPE!")

        bias = self.embeddings(rpe)
        bias = rearrange(bias, 'm n h -> 1 h m n')

        return bias


class FullAttention(nn.Module):
    def __init__(self, d_q, d_k, d_v, d_model, h, dropout):
        super(FullAttention, self).__init__()
        """
        d_q, d_k, d_v : dimensions of input q,k,v
        d_model: dimension of hidden state in attention calculation
        h: head num
        dropout: output dropout rate
        """
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.h = h

        self.W_q = nn.Conv1d(d_q, d_model, 1)
        self.W_k = nn.Conv1d(d_k, d_model, 1)
        self.W_v = nn.Conv1d(d_v, d_model, 1)
        self.W_out = nn.Conv1d(d_model, d_v, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, input_mask, return_attn=False):
        """
        q: (B, d_q, L)
        k: (B, d_k, L)
        v: (B, d_v, L)
        input_mask: (B, L), float, 0 or 1, and 0 is padding mask
        return: out: (B, d_v, L)
                attn: (B, H, L, L)
        """

        B, _, L = q.size()

        q = self.W_q(q)  # (B, d_model, L)
        k = self.W_k(k)  # (B, d_model, L)
        v = self.W_v(v)  # (B, d_model, L)

        q = q.reshape(B, self.h, -1, L)
        k = k.reshape(B, self.h, -1, L)
        v = v.reshape(B, self.h, -1, L)

        input_mask = input_mask.bool().unsqueeze(1)  # (B, 1, L)
        total_mask = ~ input_mask

        out, attn = scalar_dot_attn(q, k, v, mask=total_mask, return_attn=return_attn)

        out = self.W_out(self.dropout(out))  # * input_mask

        return out, attn


class LocalAttention(nn.Module):
    def __init__(self, d_q, d_k, d_v, d_model, h, dropout, rpe=None):
        super(LocalAttention, self).__init__()
        """
        d_q, d_k, d_v : dimensions of input q,k,v
        d_model: dimension of hidden state in attention calculation
        h: head num
        dropout: output dropout rate
        """
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.h = h

        self.W_q = nn.Conv1d(d_q, d_model, 1)
        self.W_k = nn.Conv1d(d_k, d_model, 1)
        self.W_v = nn.Conv1d(d_v, d_model, 1)
        self.W_out = nn.Conv1d(d_model, d_v, 1)
        self.dropout = nn.Dropout(dropout)
        
        self.rpe = rpe

    def forward(self, q, k, v, input_mask, l_seg, window_size, return_attn=False):
        """
        q: (B, d_q, L)
        k: (B, d_k, L)
        v: (B, d_v, L)
        input_mask: (B, L), float, 0 or 1, and 0 is padding mask
        l_seg, window_size: hyper parameters for efficient local attention caculation
        return_attn
        return: out: (B, d_v, L)
                attn: (B, H, L, L)
        """

        # print("l_seg: ", l_seg)
        # print("window: ", window_size)

        B, _, L = q.size()
        device = q.device

        q = self.W_q(q)  # (B, d_model, L)
        k = self.W_k(k)  # (B, d_model, L)
        v = self.W_v(v)  # (B, d_model, L)

        # if L > l_seg, you need to segment the q, k, v into several segments to speed up the local attention caculation
        # else, you can directly do the local attention caculation based on the q, k, v
        if L > l_seg:  # and B != 1:

            n_seg = L // l_seg  # segment num

            # padding for the last segment
            if L % l_seg != 0:
                q = torch.cat([q, torch.zeros((B, self.d_model, l_seg - L % l_seg), device=device)], dim=-1)
                k = torch.cat([k, torch.zeros((B, self.d_model, l_seg - L % l_seg), device=device)], dim=-1)
                v = torch.cat([v, torch.zeros((B, self.d_model, l_seg - L % l_seg), device=device)], dim=-1)
                n_seg += 1

            # padding at the start and end of the key and value
            k = torch.cat([torch.zeros((B, self.d_model, window_size // 2), device=device), k,
                           torch.zeros((B, self.d_model, window_size // 2), device=device)], dim=-1)
            v = torch.cat([torch.zeros((B, self.d_model, window_size // 2), device=device), v,
                           torch.zeros((B, self.d_model, window_size // 2), device=device)], dim=-1)

            # segment and multi-head for efficient local attention caculation
            q_seg = rearrange(q, 'b (h d) (n l) -> (b n) h d l', n=n_seg, h=self.h)  # (B, H*D, N*l_q) -> (B*N, H, D, l_q), where l_q = L_seg
            k_seg = k.unfold(2, (l_seg + 2 * (window_size // 2)), l_seg)  # (B, D, N, l_k) , where l_k = l_seg + 2 * (window_size // 2)
            k_seg = rearrange(k_seg, 'b (h d) n l -> (b n) h d l', h=self.h)  # (B*N, H, D, l_k)
            v_seg = v.unfold(2, (l_seg + 2 * (window_size // 2)), l_seg)  # (B, D, N, l_k) , where l_k = l_seg + 2 * (window_size // 2)
            v_seg = rearrange(v_seg, 'b (h d) n l -> (b n) h d l', h=self.h)  # (B*N, H, D, l_k)

            # window mask generation
            window_mask = create_window_mask(l_seg, window_size)  # (l_q, l_k)
            window_mask = window_mask.to(device)

            # key padding mask generation
            if L % l_seg != 0:
                mask = torch.cat([input_mask, torch.zeros((B, l_seg - L % l_seg), device=device)], dim=-1)
            else:
                mask = input_mask

            mask = torch.cat([torch.zeros((B, window_size // 2), device=device), mask, 
                              torch.zeros((B, window_size // 2), device=device)], dim=-1)
            mask = mask.unfold(1, (l_seg + 2 * (window_size // 2)), l_seg)  # (B, N, l_k)
            mask = mask.reshape(-1, 1, l_seg + 2 * (window_size // 2)).bool()  # (B*N, 1, l_k)

            # total mask generation
            total_mask = ~ (window_mask & mask)  # (B*N, l_q, l_k)

            if self.rpe is not None:
                rpe_bias = self.rpe.compute_bias(l_seg, l_seg + 2 * (window_size // 2))
                out, attn = scalar_dot_attn(q_seg, k_seg, v_seg, mask=total_mask, rpe=rpe_bias, return_attn=return_attn)  # (B*N, D, l_q)
            else:
                out, attn = scalar_dot_attn(q_seg, k_seg, v_seg, mask=total_mask, return_attn=return_attn)  # (B*N, D, l_q)

            # reshape output
            out = rearrange(out, '(b n) d l -> b d (n l)', n=n_seg)
            out = self.W_out(self.dropout(out[..., :L]))  # * input_mask  # (B, d_model, L) -> (B, d_v, L)
            
            if attn is not None:
                attn_distrib = extract_dis_from_attention(attn, window_size)
                attn_distrib = rearrange(attn_distrib, '(b n) h l w-> b h (n l) w', n=n_seg)
                attn_distrib = attn_distrib[..., :L, :]  # (B, H, L, window_size)
            else:
                attn_distrib = None

        else:

            q = q.reshape(B, self.h, -1, L)
            k = k.reshape(B, self.h, -1, L)
            v = v.reshape(B, self.h, -1, L)

            window_mask = create_window_mask(L, window_size)
            begin_index = window_size // 2
            window_mask = window_mask[:, begin_index:L+begin_index]  # (L, L)
            window_mask = window_mask.to(device)

            mask = input_mask.bool().unsqueeze(1)  # (B, 1, L)

            total_mask = ~ (window_mask & mask)  # (B, L, L)

            if self.rpe is not None:
                rpe_bias = self.rpe.compute_bias(L, L)
                out, attn = scalar_dot_attn(q, k, v, mask=total_mask, rpe=rpe_bias, return_attn=return_attn)
            else:
                out, attn = scalar_dot_attn(q, k, v, mask=total_mask, return_attn=return_attn)
            
            # reshape output
            out = self.W_out(self.dropout(out))  # * input_mask  # (B, d_model, L) -> (B, d_v, L)

            if attn is not None:
                attn_distrib = extract_dis_from_attention(attn, window_size)
            else:
                attn_distrib = None

        return out, attn_distrib  # (B, H, L, window_size)
