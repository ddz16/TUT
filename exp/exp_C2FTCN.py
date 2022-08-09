import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from loguru import logger
import numpy as np
from tqdm import tqdm

from models.ASFormer import ASFormer
from models.C2FTCN import C2F_TCN

from data_provider.data_factor import data_provider

from eval import segment_bars_with_confidence


class ExpC2FTCN:
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

        model = C2F_TCN(self.configs)

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
            correct = 0
            total = 0

            for i, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):
                batch_input = batch_data['feature'].to(self.device)
                batch_target = batch_data['label'].to(self.device)
                mask = batch_data['mask'].to(self.device)

                optimizer.zero_grad()
                prediction = self.model(batch_input, mask)
                prediction = torch.log(prediction + 1e-10)

                loss = torch.tensor(0.0).to(self.device)

                loss += self.ce(prediction.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                loss += self.configs.gamma * torch.mean(
                    torch.clamp(
                        self.mse(
                            p[:, :, 1:],
                            p.detach()[:, :, :-1]),
                        min=0, max=16) * mask.unsqueeze(1)[..., 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(prediction, 1)
                correct += ((predicted == batch_target).float() * mask).sum().item()
                total += torch.sum(mask).item()

            scheduler.step(epoch_loss)
            torch.save(self.model.state_dict(), self.configs.model_dir + "/epoch-" + str(epoch + 1) + ".model")
            # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

            logger.info("[epoch %d]: epoch loss = %f, acc = %f" % (epoch + 1, epoch_loss / len(train_data),
                                                                     float(correct)/total))
            if (epoch + 1) % 10 == 0:
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

                prediction = self.model(batch_input, mask)
                _, predicted = torch.max(prediction, 1)
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
                print("predict video id {}, [length: {}]".format(vid, length))
                
                prediction = self.model(batch_input, mask)

                confidence, predicted = torch.max(F.softmax(prediction, dim=1).data, 1)
                confidence, predicted = confidence.squeeze(), predicted.squeeze()  # (L)

                batch_target = batch_target.squeeze()  # (L)

                segment_bars_with_confidence(self.configs.results_dir + '/{}.png'.format(vid),
                                                confidence.tolist(),
                                                batch_target.tolist(), predicted.tolist())

                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))

                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(self.configs.results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()


if __name__ == '__main__':
    pass