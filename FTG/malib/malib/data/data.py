import collections
import random
from typing import Dict, List

from torch.utils.data import DataLoader

from malib.data import MyDataset


class Data(object):
    """
    standard data class

    the Data class Provides uniform interfaces to operating datas
    """

    def __init__(self, config) -> None:
        """init Data object with the global config

        Args:
            config (BaseConfig): global config object
        """
        super().__init__()
        self.config = config
        self.train_data_num = config.data_config.train_data_num
        self.tra_len = config.data_config.tra_len
        self.data_to_save = list(config.data_config.data_to_save.player_data)
        self.data_to_save += config.data_config.data_to_save.other_data
        self.data_to_save += config.data_config.data_to_save.preprocessing_data
        self.max_data_len = config.data_config.data_capacity
        self.data_sample_mode = config.data_config.data_sample_mode
        data_sample_methods = {
            "USWR": self.sample_datas_USWR,
            "FIFO": self.sample_datas_FIFO,
        }
        self._sample_datas = data_sample_methods[self.data_sample_mode]

        for data_name in self.data_to_save:
            self.__dict__[data_name] = []

    def clear(self):
        """clear all data in the Data object"""
        for k in self.data_to_save:
            self.__dict__[k].clear()

    def set_data(self, key, value):
        self.__dict__[key] = value

    def append(self, data: Dict) -> None:
        """add data to Data object

        similar to append of list

        Args:
            data (Dict): data dict like {'feature':f_data,'action':action_data}
        """
        for k, v in data.items:
            assert k in self.data_to_save
            self.__dict__[k].append(v)

    def extend(self, Data_list: List) -> None:
        """concat two or more Data object  to one

        similar to extent of list

        Args:
            Data_list (list): list of Data object
        """
        for Data_obj in Data_list:
            for k in self.data_to_save:
                self.__dict__[k].extend(Data_obj.__dict__[k])

    def add_data(self, data_dict: Dict) -> None:
        """add data from data_storage to Data

        Args:
            data_dict (Dict): data dict,{'feature':feature_data,'action':action_data}
        """
        for k, v in data_dict.items():

            self.__dict__[k].extend(v)
        if self.count > self.max_data_len:
            for k in self.data_to_save:
                self.__dict__[k] = self.__dict__[k][-self.max_data_len :]

    def __len__(self):
        return self.count

    @property
    def count(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        c = -1
        for k in self.data_to_save:
            c = len(self.__dict__[k])
            break
        return c

    def get_data(self, data_num: int = None, key: List[str] = None):
        """get data from Data using aspecified data sampling method,ie. FIFO

        Data class provide different data sampling methods to get data

        Args:
            data_num (int, optional): how many data to sample. Defaults to None.
            key (List[str], optional): what kind of data to sample. Defaults to None.

        Returns:
            Data: Data object
        """
        data_len = data_num if data_num else self.train_data_num
        assert data_len <= self.count
        data = Data(self.config)
        data_dict = self._sample_datas(data_len)
        data.__dict__.update(data_dict)
        return data

    def sample_datas_FIFO(self, data_len: int) -> Dict:
        """sample data by first in first out method

        Args:
            data_len (int): [description]

        Returns:
            Dict: data dict
        """
        data_dict = {}
        for k in self.data_to_save:
            v = self.__dict__[k]
            d = v[:data_len]
            self.__dict__[k] = v[data_len:]
            data_dict[k] = d
        return data_dict

    def sample_datas_USWR(self, data_len):
        """有放回无序采样

        Args:
            data_len (int): data number

        Returns:
            dict: data dict
        """
        indexes = random.sample(range(self.count), data_len)
        data_dict = collections.defaultdict(lambda: [])
        for ind in indexes:
            for d_k in self.data_to_save:
                data_dict[d_k].append(self.__dict__[d_k][ind])
        return data_dict

    def make_dataset(self, data_list):
        """make pytorch DataLoader from the the list of tuple

        Args:
            data_list (list): list of tuple

        Returns:
            DataLoader: DataLoader object
        """
        batch_size = self.config.data_config.batch_size
        data_set = MyDataset(data_list)
        data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)
        return data_loader
