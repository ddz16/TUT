'''
    Adapted from https://github.com/yabufarha/ms-tcn
'''

import torch
import numpy as np
import random
import math
import torch.nn.functional as F


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        if self.num_classes == 48:  # breakfast
            truncate_len = 200000
            min_len = 100
        elif self.num_classes == 19:  # 50salads, 17+2
            truncate_len = 5000
            min_len = 100
        else:                         # gtea
            truncate_len = 300000
            min_len = 10

        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_target_source = []
        batch_length = []
        batch_chunk = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            features_down = features[:, ::self.sample_rate]
            classes_down = classes[::self.sample_rate]
            batch_target_source.append(classes_down)
            chunk_num = 1
            batch_length.append(np.shape(features_down)[1])
            if np.shape(features_down)[1] > truncate_len:
                chunk_num = math.ceil(float(np.shape(features_down)[1]) / truncate_len)
            if np.shape(features_down)[1] < min_len:
                chunk_num = math.ceil(min_len / float(np.shape(features_down)[1]))
                batch_chunk.append(1.0/chunk_num)
                batch_input.append(np.repeat(features_down, chunk_num, axis=1))
                batch_target.append(np.repeat(classes_down, chunk_num, axis=0))
                continue
            batch_chunk.append(chunk_num)
            batch_input.append(features_down[:, ::chunk_num])
            batch_target.append(classes_down[::chunk_num])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        # (B, D, L), (B, L), (B, C, L), [video id], [video length], [chunk num]
        return batch_input_tensor, batch_target_tensor, mask, batch, batch_length, batch_chunk, batch_target_source
