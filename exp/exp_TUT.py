import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from loguru import logger
import numpy as np
from tqdm import tqdm

from models.TUT import TUT

from data_provider.data_factor import data_provider

from eval import segment_bars_with_confidence
from utils import KL_loss, SKL_loss, JS_loss, W_loss, L2_loss, CE_loss, class2boundary, extract_dis_from_attention, create_distribution_from_cls, plot_attention_map


class ExpTUT:
    def __init__(self, configs):
        self.configs = configs
        self.device = configs.device
        self.num_classes = configs.num_classes
        self.model = self._build_model(configs.model).to(configs.device)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')

        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))
        # logger.add('logs/' + args.dataset + "_" + args.split + "_{time}.log")
        # logger.add(sys.stdout, colorize=True, format="{message}")
    
    def _build_model(self, model_type):

        model = TUT(self.configs)

        return model

    def _get_data(self, mode):
        data_set, data_loader = data_provider(self.configs, mode)
        return data_set, data_loader

    def train(self):
        train_data, train_loader = self._get_data(mode='train')
        _, test_loader = self._get_data(mode='test')

        optimizer = optim.Adam(self.model.parameters(), lr=self.configs.lr, betas=(0.9, self.configs.adambeta), weight_decay=self.configs.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        self.model.train()

        for epoch in range(self.configs.num_epochs):
            epoch_loss = 0
            epoch_ba_loss = 0
            correct = 0
            total = 0

            for i, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):
                batch_input = batch_data['feature'].to(self.device)
                batch_target = batch_data['label'].to(self.device)
                mask = batch_data['mask'].to(self.device)

                optimizer.zero_grad()
                predictions, all_attns = self.model(batch_input, mask)

                loss = torch.tensor(0.0).to(self.device)

                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += self.configs.gamma * torch.mean(
                        torch.clamp(
                            self.mse(
                                F.log_softmax(p[:, :, 1:], dim=1),
                                F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                            min=0, max=16) * mask.unsqueeze(1)[..., 1:])

                if self.configs.baloss:
                    baloss = torch.tensor(0.0).to(self.device)
                    use_chi = False
                    loss_layer_num = 1
                    #    (1,L)      (begin_length)   (end_length)
                    # extract from all layers (different resolution) to get begin_index and end_index
                    _, begin_index, end_index = class2boundary(batch_target)
                    down_target = batch_target
                    begin_index_list = [begin_index]
                    end_index_list = [end_index]
                    B, _, L = batch_input.shape
                    # print(L)
                    len_list = [L // (2 ** i) for i in range(loss_layer_num + 1)]  # [L, L/2, L//4, ...]
                    for i in range(loss_layer_num):
                        down_target = F.interpolate(down_target.float().unsqueeze(0), size=len_list[i+1]).squeeze(0).long()
                        _, begin_index, end_index = class2boundary(down_target)
                        begin_index_list.append(begin_index)
                        end_index_list.append(end_index)

                    for attn in all_attns:  # each attn is each stage list
                        # attn: a list of (B, H, L, window_size)
                        for i in range(loss_layer_num):
                            # print(begin_index_list[i+1])
                            if begin_index_list[i+1].shape[0] > 0 and end_index_list[i+1].shape[0] > 0:
                                attn_begin = torch.index_select(attn[i], dim=2, index=begin_index_list[i+1].to(self.device))  # (B,H,l,window_size), encoder layer attn begin
                                attn_end = torch.index_select(attn[i], dim=2, index=end_index_list[i+1].to(self.device))  # (B,H,l,window_size), encoder layer attn end
                                baloss += self.configs.beta * KL_loss(attn_begin, create_distribution_from_cls(0, self.configs.window_size, use_chi).to(self.device))
                                baloss += self.configs.beta * KL_loss(attn_end, create_distribution_from_cls(2, self.configs.window_size, use_chi).to(self.device))
                            # print(attn_begin)
                            # print(attn_end)
                            # print(baloss)

                            # attn_begin = torch.index_select(attn[-i-1], dim=2, index=begin_index_list[i].to(device))  # (1,H,l,window_size), decoder layer attn begin
                            # attn_end = torch.index_select(attn[-i-1], dim=2, index=end_index_list[i].to(device))  # (1,H,l,window_size), decoder layer attn begin
                            # baloss += self.configs.beta * KL_loss(attn_begin, create_distribution_from_cls(0, self.configs.window_size, use_chi).to(device))
                            # baloss += self.configs.beta * KL_loss(attn_end, create_distribution_from_cls(2, self.configs.window_size, use_chi).to(device))
                        # break # comment on 50Salads and GTEA, meaning use all stages; if not comment, meaning only use prediction stage
                    epoch_ba_loss += baloss.item()
                    loss += baloss

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions.data[-1], 1)
                correct += ((predicted == batch_target).float() * mask).sum().item()
                total += torch.sum(mask).item()

            scheduler.step(epoch_loss)
            torch.save(self.model.state_dict(), self.configs.model_dir + "/epoch-" + str(epoch + 1) + ".model")
            # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            if self.configs.baloss:
                logger.info("[epoch %d]: epoch loss = %f, ba loss = %f, acc = %f" % (epoch + 1, epoch_loss / len(train_data), epoch_ba_loss / len(train_data),
                                                                     float(correct)/total))
            else:
                logger.info("[epoch %d]: epoch loss = %f, acc = %f" % (epoch + 1, epoch_loss / len(train_data),
                                                                     float(correct)/total))

            if (epoch + 1) % 1 == 0:
                self.test(test_loader, epoch)


    def test(self, test_loader, epoch):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for i, batch_data in tqdm(enumerate(test_loader), total=len(test_loader)):
                batch_input = batch_data['feature'].to(self.device)
                batch_target = batch_data['label'].to(self.device)
                mask = batch_data['mask'].to(self.device)

                predictions, _ = self.model(batch_input, mask)
                _, predicted = torch.max(predictions.data[-1], 1)
                correct += ((predicted == batch_target).float() * mask).sum().item()
                total += torch.sum(mask).item()

        acc = float(correct) / total
        logger.info("---[epoch %d]---: test acc = %f" % (epoch + 1, acc))
        self.model.train()


    def predict(self):
        test_data, test_loader = self._get_data(mode='test')
        actions_dict = test_data.__get_actions_dict__()
        sample_rate = test_data.__get_sample_rate__()

        self.model.eval()
        with torch.no_grad():
            self.model.load_state_dict(torch.load(self.configs.model_dir + "/epoch-" + str(self.configs.num_epochs) + ".model"))
            for i, batch_data in enumerate(test_loader):
                assert len(batch_data['id']) == 1
                batch_input = batch_data['feature'].to(self.device)
                batch_target = batch_data['label'].to(self.device)
                mask = batch_data['mask'].to(self.device)
                vid = batch_data['id'][0]
                length = batch_data['length'][0]
                # print("predict video id {}, [length: {}]".format(vid, length))
                
                predictions, all_attns = self.model(batch_input, mask)

                for i in range(len(predictions)):
                    confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
                    confidence, predicted = confidence.squeeze(), predicted.squeeze()  # (L)

                    batch_target = batch_target.squeeze()  # (L)

                    segment_bars_with_confidence(self.configs.results_dir + '/{}_stage{}.png'.format(vid, i),
                                                 confidence.tolist(),
                                                 batch_target.tolist(), predicted.tolist())

                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))

                if vid == 'rgb-27-1.txt':
                    for i in range(len(all_attns)):
                        plot_attention_map(all_attns[i][0][0,0,100:600,:].cpu().numpy(), self.configs.attn_dir + '/{}_stage{}_encoder.png'.format(vid, i))
                        # plot_attention_map(all_attns[i][0][0,1,100:600,:].cpu().numpy(), self.configs.attn_dir + '/{}_stage{}_encoder.png'.format(vid, 1))

                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(self.configs.results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()


if __name__ == '__main__':
    pass