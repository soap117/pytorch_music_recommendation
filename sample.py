import os
import torch
import torch.nn
import numpy as np
class music_data:
    def __init__(self, data_path):
        self.data_path = data_path
        self.file_list = os.listdir(self.data_path)
        self.name2id = {}
        self.id2name = {}
        count = 0
        self.index = 0
        for file in self.file_list:
            self.name2id[file] = count
            self.id2name[count] = file
            count += 1


    def get_batch(self, batch_index):
        mfcc_features = []
        for ith in batch_index:
            mfcc_feature = np.load(os.path.join(self.data_path, self.id2name[ith]))
            mfcc_features.append(mfcc_feature)
        mfcc_features = torch.Tensor(np.array(mfcc_features)).cuda()
        return mfcc_features
