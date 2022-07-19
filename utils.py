from torch.distributions.normal import Normal
from torch.distributions.chi2 import Chi2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from einops import repeat


def class2boundary(target):
    """
    :param target: (B, L)
    :return: (B, L), begin:0 , middle:1, end:2
    """
    target = target.cpu().numpy()
    B, T = target.shape
    assert B == 1

    boundary = np.zeros((B, T), dtype=np.int64)
    begin_list = []
    end_list = []

    for b in range(B):
        boundary[b, 0] = 0
        for t in range(1, T - 1):
            if target[b, t] != target[b, t - 1]:
                boundary[b, t] = 0
                begin_list.append(t)
            else:
                if target[b, t] == target[b, t + 1]:
                    boundary[b, t] = 1
                else:
                    boundary[b, t] = 2
                    end_list.append(t)
        boundary[b, T - 1] = 2
    # print(boundary)
    boundary = torch.from_numpy(boundary)
    begin = torch.tensor(begin_list)
    end = torch.tensor(end_list)
    return boundary, begin, end


# def downsmaple_boundary(boundary, size=2):
#     """
#     :param boundary: (B,L), begin:0 , middle:1, end:2
#     :param size: down smaple rate
#     :return: (B,L//size)
#     """
#     boundary = boundary.numpy()
#     B, T = boundary.shape
#     assert B == 1
#
#     boundary_down = np.zeros((B, T//size), dtype=np.int64)
#     begin_list = []
#     end_list = []
#
#     for b in range(B):
#         for index, t in enumerate(range(0, T - 1, size)):
#             if 0 in boundary[b, t:(t + size)]:
#                 boundary_down[b, index] = 0
#                 begin_list.append(index)
#             elif 2 in boundary[b, t:(t + size)]:
#                 boundary_down[b, index] = 2
#                 end_list.append(index)
#             else:
#                 boundary_down[b, index] = 1
#     # print(boundary)
#     boundary_down = torch.from_numpy(boundary_down)
#     begin = torch.tensor(begin_list)
#     end = torch.tensor(end_list)
#     return boundary_down, begin, end


# def create_distribution_from_boundary(boundary, window_size):
#     """
#     :param boundary: (B,L)
#     :param window_size:
#     :return: (B,L,window_size)
#     """
#     boundary = boundary.numpy()
#     B, L = boundary.shape
#     boundary = boundary[:, (window_size // 2): (L - window_size // 2)]
#     _, new_L = boundary.shape
#     dis = np.ones((B, new_L, window_size)) / window_size
#
#     for b in range(B):
#         for t in range(new_L):
#             if boundary[b][t] == 0:
#                 dis[b][t] = create_chi2_distribution(window_size, right=True)
#             elif boundary[b][t] == 1:
#                 dis[b][t] = create_normal_distribution(10, window_size)
#             elif boundary[b][t] == 2:
#                 dis[b][t] = create_chi2_distribution(window_size, right=False)
#
#     dis = torch.from_numpy(dis)
#     return dis


def create_distribution_from_cls(cls, window_size):
    """
    cls: 0 is begin, 1 is middle, 2 is end
    return (window_size)
    """
    if cls == 0:
        dis = create_chi2_distribution(window_size, right=True)
    elif cls == 2:
        dis = create_chi2_distribution(window_size, right=False)
    else:
        dis = create_normal_distribution(10, window_size)
    return dis


def create_normal_distribution(scale, window_size):
    norm_distribution = Normal(0, scale)
    point_index = torch.arange(-(window_size//2), window_size//2+1)
    dis = torch.exp(norm_distribution.log_prob(point_index))
    dis = dis / torch.sum(dis)
    return dis


def create_uniform_distribution(window_size):
    return torch.ones(window_size) / window_size


def create_chi2_distribution(window_size, right=True):
    """
    right: right part has higher similarity
    """
    chi2_distribution = Chi2(4)
    point_index = 2.0 / ((window_size+1)//2) * torch.arange(-(window_size//2), window_size//2+1) + 2
    dis = torch.exp(chi2_distribution.log_prob(point_index))
    dis = dis / torch.sum(dis)
    if right:
        return dis
    else:
        return dis[range(window_size-1, -1, -1)]


def create_half_distribution(window_size, right=True):
    """
    right: right part has higher similarity
    """
    dis = torch.zeros(window_size)
    dis[-window_size//2:] = 1 / (window_size//2+1)
    if right:
        return dis
    else:
        return dis[range(window_size-1, -1, -1)]


def extract_dis_from_attention(attention_map, window_size):
    """
    :param attention_map: [B, H, L_seg , (L_seg + 2 * window_size // 2)] or [B, H, L, L]
    :return: [B, H, L_seg, window_size]
    """
    B, H, l1, l2 = attention_map.shape
    if l1 == l2:
        attention_map = torch.cat([torch.zeros(l1, window_size // 2, device=attention_map.device),
                                   attention_map,
                                   torch.zeros(l1, window_size // 2, device=attention_map.device)], dim=1)
        l2 = attention_map.shape[3]

    assert attention_map.shape[2] + 2 * (window_size // 2) == attention_map.shape[3]
    dis =  torch.as_strided(attention_map, (B, H, l1, window_size), (H*l1*l2, l1*l2, l2+1, 1))
    return dis


def KL_loss(scores_dis, dis):
    """
    Kullback Leibler Divergence
    :param scores_dis: (B, H, L, window_size)
    :param dis: (window_size)
    :return: scalar
    """
    kl = dis * (torch.log(dis + 0.00001) - torch.log(scores_dis + 0.00001))
    return torch.mean(torch.sum(kl, dim=-1))


def SKL_loss(scores_dis, dis):
    """
    Symmetrized Kullback Leibler Divergence
    :param scores_dis: (B, H, L, window_size)
    :param dis: (window_size)
    :return: scalar
    """
    skl1 = dis * (torch.log(dis + 0.00001) - torch.log(scores_dis + 0.00001)) / 2
    skl2 = scores_dis * (torch.log(scores_dis + 0.00001) - torch.log(dis + 0.00001)) / 2
    return torch.mean(torch.sum(skl1 + skl2, dim=-1))


def JS_loss(scores_dis, dis):
    """
    Jensen Shannon divergence
    :param scores_dis: (B, H, L, window_size)
    :param dis: (window_size)
    :return: scalar
    """
    m = (scores_dis + dis) / 2
    js1 = scores_dis * (torch.log(scores_dis + 0.00001) - torch.log(m + 0.00001)) / 2
    js2 = dis * (torch.log(dis + 0.00001) - torch.log(m + 0.00001)) / 2
    return torch.mean(torch.sum(js1 + js2, dim=-1))


def W_loss(scores_dis, dis):
    """
    Wasserstein distance
    :param scores_dis: (B, H, L, window_size)
    :param dis: (window_size)
    :return: scalar
    """

    # make cdf with cumsum
    cdf_tensor_a = torch.cumsum(scores_dis, dim=-1)
    cdf_tensor_b = torch.cumsum(dis, dim=-1)
    return torch.mean(torch.sum(torch.abs(cdf_tensor_a-cdf_tensor_b), dim=-1))


def L2_loss(scores_dis, dis):
    """
    L2 distance (MSE)
    :param scores_dis: (B, H, L, window_size)
    :param dis: (window_size)
    :return: scalar
    """

    return torch.mean(torch.sum((scores_dis - dis) ** 2, dim=-1)) / 2


def CE_loss(scores_dis, dis):
    """
    Cross Entropy
    :param scores_dis: (B, H, L, window_size)
    :param dis: (window_size)
    :return: scalar
    """

    return -torch.mean(torch.sum(dis * torch.log(scores_dis + 0.00001), dim=-1))



def plot_attention_map(attn, save_path):
    fig = plt.figure()
    plt.matshow(attn.T)
    plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()



