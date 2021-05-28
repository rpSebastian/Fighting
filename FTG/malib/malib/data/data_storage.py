import collections
from typing import Dict, List
import ray

from malib import data
from malib.game import game
from malib.utils import load
from .data import Data


class DataStorage(object):
    """save data from games, sample data for training"""

    def __init__(self, config: Dict, agent_list=None, register_handle=None) -> None:
        super().__init__()
        self.config = config
        self.agent_list = self.config.learn_agent_id
        if agent_list is not None:
            self.agent_list = agent_list
        self.register_handle = register_handle
        self.process_on_episode = self.setup_data_process()
        self.player_data_to_save = config.data_config.data_to_save.player_data
        self.game_data_to_save = config.data_config.data_to_save.other_data
        self.data_to_save = self.game_data_to_save + self.player_data_to_save
        self.use_game_data = True  # debug
        if "game_data" in self.game_data_to_save:
            self.use_game_data = True
        else:
            self.use_game_data = False
        self.train_data_num = config.data_config.train_data_num
        self.batch_size = config.data_config.batch_size
        self.tra_len = config.data_config.tra_len
        self.which_player = config.learn_player_id

        self.datas = collections.defaultdict(lambda: [])
        self.tra_datas = collections.defaultdict(
            lambda: collections.defaultdict(lambda: [])
        )
        self.agent_data_dict = collections.defaultdict(lambda: Data(self.config))

        self.episode_data = collections.defaultdict(
            lambda: collections.defaultdict(lambda: [])
        )
        self.game_last_info = []
        self.game_total_steps = 0

        # self.data_episode_index=0

        # self.player_data=collections.defaultdict(lambda:[])

    def setup_data_process(self):
        process_param = self.config.data_config.process_on_episode
        if type(process_param) is str:
            assert self.register_handle is not None
            result = ray.get(self.register_handle.get.remote(process_param))
            return result
        else:
            return process_param

    def clear_data(self):
        """clear all datas in datastorage"""
        self.datas = collections.defaultdict(lambda: [])
        self.tra_datas = collections.defaultdict(
            lambda: collections.defaultdict(lambda: [])
        )

    def add_episode_data(self, episode_datas: List) -> None:
        """处理游戏数据，按一局一局进行处理，将一局的数据按照agent name处理成每个agent对应的数据，放入
            self.datas中

        Args:
            episode_data (List): list contains episode data
             eg: episode data: {'game_id', 'p0', 'done', 'game_data'}
        """
        for episode_data in episode_datas:
            # self.data_episode_index+=1
            data_tmp = collections.defaultdict(
                lambda: collections.defaultdict(lambda: [])
            )
            other_data = {}
            for data_key in self.game_data_to_save:
                other_data[data_key] = episode_data[data_key]
            player_data = episode_data[self.which_player]

            # for agent_name in self.config.learn_agent_id:
            for agent_name in self.agent_list:
                for d_k in self.game_data_to_save:
                    data_tmp[agent_name][d_k] = episode_data[d_k]
            # transfer player dict data to agent dict data
            for game_step in range(len(player_data)):
                p_data = player_data[game_step]
                for data_type, data_dict in p_data.items():
                    # for agent_name in self.config.learn_agent_id:
                    for agent_name in self.agent_list:
                        if data_type == "obs":
                            data_tmp[agent_name][data_type].append(data_dict)
                        else:
                            data_tmp[agent_name][data_type].append(
                                data_dict[agent_name]
                            )

            if self.process_on_episode:
                d = self.process_on_episode(self, data_tmp)
                for agent_name, agent_data in d.items():
                    for data_name, data in agent_data.items():
                        assert data_name not in self.data_to_save, "名字重复:{}".format(
                            data_name
                        )
                        data_tmp[agent_name][data_name] = data

            for k, v in data_tmp.items():
                self.datas[k].append(v)

        self.make_tra_data()

    def make_sars_data(self):
        pass

    def make_tra_data(self):
        """make trajectory data from episode data"""
        tra_datas = collections.defaultdict(lambda: collections.defaultdict(lambda: []))
        for agent_name, agent_data in self.datas.items():
            while agent_data:
                agent_episode_data = agent_data.pop(0)

                data_list = list(agent_episode_data.values())[0]
                if len(data_list) < self.tra_len:
                    continue

                tra_dict = collections.defaultdict(lambda: [])
                # 保存数据的方法1：保存指定长度的轨迹数据、轨迹下一步数据和游戏最后一步的数据
                # TODO：添加配置项，自定义更精准的数据保存方法，只保存需要的数据，去除冗余的数据
                # 数据处理方法
                for k, v in agent_episode_data.items():
                    # k:data_type, eg: feature,action,obs, ...
                    tra_dict[k] = []
                    tra_dict[k].append(v[-1])
                    for i in range(len(v)):
                        tra_dict[k].insert(-1, v[i])
                        if len(tra_dict[k]) == self.tra_len + 1:
                            d_tmp = list(tra_dict[k])
                            if i == len(v) - 1:
                                d_tmp.insert(-1, v[i])
                            else:
                                d_tmp.insert(-1, v[i + 1])

                            tra_datas[agent_name][k].append(d_tmp)
                            tra_dict[k].pop(0)

            self.datas[agent_name] = []
        for agent_name, agent_dict_data in tra_datas.items():
            self.agent_data_dict[agent_name].add_data(agent_dict_data)

    def add_data(self, data: List) -> None:

        pass

    def add_data_ref(self, data_ref: List) -> None:
        # TODO: 一次性get是否需要改进效率
        data = ray.get(data_ref)
        self.process_data_by_episode(data)

    def process_data_by_episode(self, data_list):
        episode_datas = []
        for i, g_data_dict in enumerate(data_list):
            g_id = i + 1
            d = g_data_dict[g_id]
            for d_key, d_data in d.items():
                if d_key != "data_num":
                    self.episode_data[g_id][d_key] += d_data
                else:
                    self.game_total_steps += d_data
            while True in self.episode_data[g_id]["done"]:
                episode_data = {}
                episode_data["game_id"] = g_id
                done_index = self.episode_data[g_id]["done"].index(True)
                for k in self.episode_data[g_id].keys():
                    v = self.episode_data[g_id][k]
                    episode_data[k] = v[: done_index + 1]
                    self.episode_data[g_id][k] = v[done_index + 1 :]
                episode_datas.append(episode_data)
                last_step_info = episode_data["game_data"][-1]
                self.game_last_info.append(last_step_info)
        if episode_datas:
            self.add_episode_data(episode_datas)

    def get_game_last_info(self):
        """get the game result info

        Returns:
            Dict: game info
        """
        # game last info is use to note game result during training
        ret = self.game_last_info
        self.game_last_info = []
        return ret

    def get_data(self, agent_name: str = None, data_num: int = None) -> Dict:
        """get data from Data class

        Args:
            agent_name (str, optional): get data for specified agent,if None,
                get all agents' data. Defaults to None.
            data_num (int, optional): [data count]. if None, set the data connt
                from  config. Defaults to None.

        Returns:
            Dict: data dict like: {'a0':a0_data,'a1':a1_data}
        """
        if agent_name:
            return self.agent_data_dict[agent_name].get_data(data_num)
        else:
            d = {}
            for k, v in self.agent_data_dict.items():
                d[k] = v.get_data(data_num)
            return d

    def get_data_ref(self, agent_name: str = None, data_num: int = None):
        data = self.get_data(agent_name, data_num)
        d_ref = {}
        for k, v in data.items():
            v_ref = ray.put(v)
            d_ref[k] = v_ref
        return d_ref

    @property
    def count(self) -> int:
        """compute how many data in datastorage

        Returns:
            int: the data length of datastorage
        """
        l = 0
        for v in self.agent_data_dict.values():
            l = len(v)
            break
        return l

    @property
    def ready(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        if self.count >= self.train_data_num:
            return True
        else:
            return False

    def data_ready(self) -> bool:
        if self.count >= self.train_data_num:
            return True
        else:
            return False

    @classmethod
    def as_remote(
        cls,
        num_cpus: int = 1,
        num_gpus: int = 0,
        memory: int = None,
        object_store_memory: int = None,
        resources: dict = None,
    ) -> type:
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)
