import torch
import numpy as np
import random
import math
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate


def my_collate_func(batchs):
    features, labels, ids, lengths = zip(*batchs)
    ids = list(ids)
    lengths = list(lengths)
    batch_size = len(batchs)
    max_seq_length = max(lengths)  # max length in one batch

    mask_batch = torch.zeros((batch_size, max_seq_length))  # mask
    fea_batch = []
    label_batch = []
    for i in range(batch_size):
        fea_batch.append(torch.from_numpy(features[i].T))
        label_batch.append(torch.from_numpy(labels[i]))
        mask_batch[i, :lengths[i]] = 1

    return  dict(
        feature = pad_sequence(fea_batch, batch_first=True, padding_value = 0.).transpose(1, 2),  # (B, D, L)
        label = pad_sequence(label_batch, batch_first=True, padding_value = -100).long(),  # (B, L)
        id = ids,
        length = lengths,
        mask = mask_batch,  # (B, L)
        )


class Dataset_food(Dataset):
    def __init__(self, root="data/", dataset="50salads", split="1", mode="train"):
        self.root = root
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.list_of_examples = []
        self.actions_dict = {}
        self.reverse_dict = {}

        # use the full temporal resolution @ 15fps
        self.sample_rate = 1
        # sample input features @ 15fps instead of 30 fps
        # for 50salads, and up-sample the output to 30 fps
        if dataset == "50salads":
            self.sample_rate = 2

        self.vid_list_file = root+dataset+"/splits/"+mode+".split"+split+".bundle"
        self.features_path = root+dataset+"/features/"
        self.gt_path = root+dataset+"/groundTruth/"
        self.__read_mapping__()
        self.__read_data__()
        self.num_classes = len(self.actions_dict)  # class num of the dataset
        # self.__show_info__()

    def __read_mapping__(self):
        mapping_file = self.root+self.dataset+"/mapping.txt"
        file_ptr = open(mapping_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])
            self.reverse_dict[int(a.split()[0])] = a.split()[1]

    def __read_data__(self):
        file_ptr = open(self.vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
    
    def __show_info__(self):
        print("action dict:  ", self.actions_dict)
        print("list_of_examples:  ", self.list_of_examples)
        print("num_classes:  ", self.num_classes)
    
    def __get_actions_dict__(self):
        return self.actions_dict
    
    def __get_sample_rate__(self):
        return self.sample_rate

    def __getitem__(self, index):
        vid = self.list_of_examples[index]
        features = np.load(self.features_path + vid.split('.')[0] + '.npy')
        # print(features.shape)  # (D, L)

        file_ptr = open(self.gt_path + vid, 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros(min(np.shape(features)[1], len(content)))
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
        # print(classes.shape)  # (L)

        features_down = features[:, ::self.sample_rate]
        classes_down = classes[::self.sample_rate]
        vlength = np.shape(features_down)[1]

        # (D, L), (L), [video id], [video length]
        return features_down, classes_down, vid, vlength

    def __len__(self):
        return len(self.list_of_examples)


class Dataset_toy(Dataset):
    def __init__(self, root="data/", dataset="50salads", split="1", mode="train"):
        self.root = root
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.list_of_examples = []
        self.actions_dict = {}
        self.reverse_dict = {}
        self.sample_rate = 1
        if dataset == "50salads":
            self.sample_rate = 2
        self.vid_list_file = root+dataset+"/splits/"+mode+".split"+split+".bundle"
        self.features_path = root+dataset+"/features/"
        self.gt_path = root+dataset+"/groundTruth/"
        self.__read_mapping__()
        self.__read_data__()
        self.num_classes = len(self.actions_dict)  # class num of the dataset
        # self.__show_info__()

    def __read_mapping__(self):
        mapping_file = self.root+self.dataset+"/mapping.txt"
        file_ptr = open(mapping_file, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])
            self.reverse_dict[int(a.split()[0])] = a.split()[1]

    def __read_data__(self):
        file_ptr = open(self.vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
    
    def __show_info__(self):
        print("action dict:  ", self.actions_dict)
        print("list_of_examples:  ", self.list_of_examples)
        print("num_classes:  ", self.num_classes)
    
    def __get_actions_dict__(self):
        return self.actions_dict
    
    def __get_sample_rate__(self):
        return self.sample_rate

    def __getitem__(self, index):
        vid = self.list_of_examples[index]
        features = np.load(self.features_path + vid.split('.')[0] + '.npy')
        # print(features.shape)  # (D, L)

        file_ptr = open(self.gt_path + vid, 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros(min(np.shape(features)[1], len(content)))
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
        # print(classes.shape)  # (L)

        features_down = features[:, ::self.sample_rate]
        classes_down = classes[::self.sample_rate]
        vlength = np.shape(features_down)[1]

        # (D, L), (L), [video id], [video length]
        return features_down, classes_down, vid, vlength

    def __len__(self):
        return len(self.list_of_examples)


if __name__ == '__main__':
    mydataset = Dataset_food(root='/data1/other/EUT/data/', dataset="breakfast")
    print(mydataset[5])
    train_loader = torch.utils.data.DataLoader(
        mydataset,
        batch_size=8,
        num_workers=4,
        collate_fn=my_collate_func
    )
    for i, d in enumerate(train_loader):
        print(d['feature'].shape)
        print(d['label'].shape)
        print(d['id'])
        print(d['length'])
        print(d['mask'].shape)
