import collections
import threading
import time

import ray

from malib import makeenv
from malib.utils import load
import copy


class Game(object):
    """this class plays step by step"""

    def __init__(
        self,
        config,
        game_id=0,
        training=False,
        env_name=None,
        player_list=None,
        register_handle=None,
    ) -> None:
        """init the Game with config,game_id and training


        Args:
            config (BaseConfig): global config object
            game_id (int, optional): the game index. Defaults to 0.
            training (Bool, optional): Game if for training or eval. Defaults to None.
        """
        super().__init__()
        self.training = training
        self.config = config
        self.register_handle = register_handle
        # self.data_to_save = config.data_config.data_to_save
        # # TODO: 指定保存哪些数据
        # self.save_game_data = False
        self.env_name = None
        self.player_list = None

        self.data_num = 0
        self.get_data = None
        if self.training:
            self.data_to_save = config.data_config.data_to_save.other_data
            self.data_players = config.data_config.data_players
            if "game_data" in self.data_to_save:
                self.save_game_data = True
            else:
                self.save_game_data = False
            if "reward" in self.data_to_save:
                self.save_reward = True
            else:
                self.save_reward = False

            self.data_async = config.data_config.data_async
            if self.data_async:
                self._get_data = self.get_data_async
            else:
                self._get_data = self.get_data_sync

            self.which_player = config.learn_player_id
            self.data_store = collections.defaultdict(lambda: [])
            if env_name:
                self.env_name = env_name
            else:
                self.env_name = config.env_name
            if player_list:
                self.player_list = player_list
            else:
                self.player_list = self.config.player_config.players
        else:
            if env_name:
                self.env_name = env_name
            elif self.config.eval_config.env_name:
                self.env_name = config.eval_config.env_name
            else:
                self.env_name = config.env_name
            if player_list:
                self.player_list = player_list
            elif self.config.eval_config.players:
                self.player_list = self.config.eval_config.players
            else:
                self.player_list = self.config.player_config.players

        self.env = self.make_env(self.env_name)

        self.players = {}
        self.make_players(config)
        self.done = True
        self.game_id = game_id
        self.game_pause = False
        self.observation = None
        if self.training:
            if self.data_async:
                self.start_game()
        self.game_info = None  # game result info, last step info

        self.lock = threading.Lock()

    def reset(self):
        """reset all state of Game"""
        # TODO: improve reset method
        self.observation = self.env.reset()
        self.done = True
        self.game_info = None

    def clear_data(self):
        """ TODO:验证game的reset和数据的清理"""
        self.data_num = 0
        self.data_store = collections.defaultdict(lambda: [])

    def make_env(self, env_name):
        # if self.config.env_config.has_key(env_name):
        if env_name in self.config.env_config:
            env_params = self.config.env_config[env_name]
            new_env_params = copy.deepcopy(env_params)
            new_env_params.raw_env = ray.get(
                self.register_handle.get.remote(env_params.raw_env)
            )
            w_list = list(env_params.wrapper)
            for w_i, w_v in enumerate(w_list):
                wrapper_i = ray.get(self.register_handle.get.remote(w_v))
                new_env_params.wrapper[w_i] = wrapper_i
            return makeenv(env_name, new_env_params)
        else:
            return makeenv(env_name)

    def start_game(self):
        """run game in a threading"""
        self.t = threading.Thread(target=self.run_game)
        self.t.setDaemon(True)
        self.t.start()

    def close(self):
        """close the game threading"""
        self.t.stop()

    def env_close(self):
        self.env.close()

    def run_game(self):
        """run game step by step,"""
        # print("game runing")
        while True:
            if self.data_num > 50:
                # TODO:  解决阻塞问题
                time.sleep(0.05)
                continue
            if not self.game_pause:
                self.run_step()

    def pause(self):
        """self.game_pause tells the game threading to stop"""
        self.game_pause = True

    def start(self):
        """start game threading"""
        self.game_pause = False

    # TODO: make this method to a public api function
    # def load(self, name):
    #     import importlib

    #     mod_name, attr_name = name.split(":")
    #     mod = importlib.import_module(mod_name)
    #     fn = getattr(mod, attr_name)
    #     return fn

    # TODO: move make players to utils to be a public method
    def make_players(self, config=None, player_list=None):
        """make players of the games

        Args:
            config (BaseConfig): global config
        """
        for player_id in self.player_list:
            player_cls_str = self.config.player_config[player_id].player_name
            if type(player_cls_str) is str:
                if ":" in player_cls_str:
                    player_cls_str = "malib." + player_cls_str
                    cls = load(player_cls_str)
                else:
                    cls = ray.get(self.register_handle.get.remote(player_cls_str))
            else:
                cls = player_cls_str
            player = cls(
                player_id,
                self.config,
                training=self.training,
                register_handle=self.register_handle,
            )
            self.players[player_id] = player

    def make_player(self, player_id):
        player_cls_str = self.config.player_config[player_id].player_name
        if type(player_cls_str) is str:
            if ":" in player_cls_str:
                player_cls_str = "malib." + player_cls_str
                cls = load(player_cls_str)
            else:
                cls = ray.get(self.register_handle.get.remote(player_cls_str))
        else:
            cls = player_cls_str
        player = cls(
            player_id,
            self.config,
            training=self.training,
            register_handle=self.register_handle,
        )
        return player

    def delete_player(self, p_id):
        assert p_id in self.player_list, "player id was not in game"
        self.player_list.remove(p_id)
        self.players.pop(p_id)

    def add_player(self, player):
        if type(player) is str:
            assert player not in self.player_list, "player id was already in game"
            player = self.make_player(player)
            self.players[player] = player
            self.player_list.append(player)
        else:
            self.players[player.player_id] = player
            self.player_list.append(player.player_id)

    def run_step(self):
        """run one step of the game"""
        if self.done:
            self.reset()
            for p in self.players.values():
                p.reset()
        actions = {}
        assert self.observation is not None
        obs_old = self.observation
        for player_k, player in self.players.items():
            action = player.select_action(self.observation[player_k])
            actions[player_k] = action
        self.observation, reward, self.done, info = self.env.step(actions)
        self.game_info = info
        if self.training:
            game_data = {}
            game_data["observation"] = obs_old
            game_data["next_observation"] = self.observation
            game_data["reward"] = reward
            game_data["done"] = self.done
            game_data["info"] = info
            self.lock.acquire()
            for player_k, player in self.players.items():
                # if player_k != self.which_player:
                #     continue
                if player_k not in self.data_players:
                    continue
                player_data = player.get_data()
                self.data_store[player_k].append(
                    # [player_data, self.observation, reward, self.done, info]
                    player_data
                )
            self.data_store["done"].append(self.done)
            self.data_store["data_num"] = len(self.data_store["done"])
            # if "game_data" in self.data_to_save:
            #     self.data_store["game_data"].append(game_data)
            if self.save_game_data:
                self.data_store["game_data"].append(game_data)
            if self.save_reward:
                self.data_store["reward"].append(reward)
            self.data_num += 1
            self.lock.release()

    def run_episode(self):
        """run one episode of the game.

        Returns:
            Dict: the game runing infos
        """
        self.done = True
        while True:
            self.run_step()
            if self.done:
                break
        if not self.training:
            if "error" in self.game_info:
                return self.run_episode()
            else:
                return self.game_info

    def get_episode_data(self):
        pass

    # def get_data(self, player_id=None, step_num=None):
    #     """get the data generated by game

    #     if data async, get the saved data
    #     if data sync, run one step game and return the data

    #     Args:
    #         player_id (str, optional): [description]. Defaults to 0.
    #         step_num (int, optional): [description]. Defaults to None.

    #     Returns:
    #         [type]: [description]
    #     """
    #     if self.data_async:
    #         # data=self.data_store[player_index]
    #         # self.data_store[player_index]=[]
    #         self.lock.acquire()
    #         data = self.data_store
    #         self.data_store = collections.defaultdict(lambda: [])
    #         self.lock.release()
    #         data_dict = {}
    #         data_dict[self.game_id] = data
    #         self.data_num = 0
    #         return data_dict
    #     else:
    #         self.run_step()
    #         data = self.data_store
    #         self.data_store = collections.defaultdict(lambda: [])
    #         data_dict = {}
    #         data_dict[self.game_id] = data
    #         self.data_num = 0
    #         return data_dict
    def get_data(self):
        return self._get_data()

    def get_data_async(self):
        # data=self.data_store[player_index]
        # self.data_store[player_index]=[]
        self.lock.acquire()
        data = self.data_store
        self.data_store = collections.defaultdict(lambda: [])
        self.lock.release()
        data_dict = {}
        data_dict[self.game_id] = data
        self.data_num = 0
        return data_dict

    def get_data_sync(self, step_num=1):
        for i in range(step_num):
            self.run_step()
        data = self.data_store
        self.data_store = collections.defaultdict(lambda: [])
        data_dict = {}
        data_dict[self.game_id] = data
        self.data_num = 0
        return data_dict

    def set_weights(self, weights, player_id, model_id):
        # TODO: change name,
        self.players[player_id].set_weights(weights, model_id)
        return "done"

    def set_ref_weights(self, weights, player_id, agent_id):
        self.players[player_id].set_ref_weights(weights, agent_id)
        return "done"

    def set_player(self, player):
        """set player in the game

        Args:
            player (BasePlayer): player
        """
        self.players[player.player_id] = player

    def update_player(self, player):
        pass

    def set_player_weights(self, player_id, player_weights):
        """set the agent model weights of player

        Args:
            player_id (str): the id of player
            player_weights (dict): dict contrains the model weights of agents in player,
                {'a0':weights_dict,'a0':weights_dict}
        """
        for agent_id, agent_weights in player_weights.items():
            self.set_weights(
                agent_weights,
                player_id,
                agent_id,
            )

    def get_epsilon(self):
        return list(self.players.values())[0].action.epsilon

    def reset_env(self, env_id=None, env=None):
        """change the env in game

        .. Note::
            Select one of two params to use
        Args:
            env_id (str): game can be game_id or BaseEnv object
            env (BaseEnv): env object
        """

        if env_id:
            self.env = makeenv(env_id)
        elif env:
            self.env = env
        else:
            raise ValueError("must give the env_id or env object to this function")

    def reset_player(self, player_id_list=None, player=None):
        """reset the player in game

        .. Note::
            players should be adaptive to the env. if game env had been reset,
            should check if the players in game adaptive to the new env

        .. TODO::
            随意设置player,目前只能删除掉一些player

        Args:
            player_id_list (List, optional): player id list tells which players to be used in game. Defaults to None.
            player (BasePlayer, optional): player give the player object used in the game. Defaults to None.

        Raises:
            ValueError: [description]
        """

        if player_id_list:
            for p_id in self.player_list:
                if p_id not in player_id_list:
                    self.delete_player(p_id)
            for p_id in player_id_list:
                if p_id not in self.player_list:
                    self.add_player(p_id)
        elif player:
            assert (
                player.player_id in self.player_list
            ), "player must be already in game"
            if player.player_id in self.player_list:
                self.set_player(player)
        else:
            raise ValueError("params can not be all None")

    @classmethod
    def as_remote(
        cls,
        num_cpus: int = 1,
        num_gpus: int = 0,
        memory: int = None,
        object_store_memory: int = None,
        resources: dict = None,
    ) -> type:
        """[summary]

        Args:
            num_cpus (int, optional): [description]. Defaults to 1.
            num_gpus (int, optional): [description]. Defaults to 0.
            memory (int, optional): [description]. Defaults to None.
            object_store_memory (int, optional): [description]. Defaults to None.
            resources (dict, optional): [description]. Defaults to None.

        Returns:
            type: [description]
        """
        return ray.remote(
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            memory=memory,
            object_store_memory=object_store_memory,
            resources=resources,
        )(cls)
