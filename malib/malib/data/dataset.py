import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        # self.data_name_list = data_name_list
        self.data_len = len(data[0])
        # for data_name in data_name_list:
        # self.__dict__[data_name] = data.datas[data_name]
        self.data = data

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        data = []
        for data_list in self.data:
            data.append(data_list[index])
        return data
