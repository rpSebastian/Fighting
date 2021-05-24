import collections
import math

import ray

from malib import makeenv
from malib.aienv.register import make
from malib.game import Game


class VectorGame(object):
    """VectorGame is remote games set used for generate data and eval policy"""

    def __init__(
        self,
        config,
        training=True,
        eval_game_num=None,
        env_name=None,
        player_list=None,
        register_handle=None,
    ) -> None:
        """init the VectorGame

        Args:
            config (BaseConfig): [description]
            training (bool, optional): [description]. Defaults to True.
            eval_game_num (int, optional): [description]. Defaults to None.
        """
        super().__init__()
        print("vectorgame init ...")
        self.config = config
        self.register_handle = register_handle
        # self.env = makeenv(config["env_name"])
        # self.players = self.make_players(config)
        self.training = training
        self.env_name = env_name
        self.player_list = player_list
        if training:
            self.game_num = config.game_number
        else:
            self.game_num = config.eval_config.eval_game_number

        self.local_game = Game(
            config=self.config,
            game_id=0,
            training=self.training,
            env_name=self.env_name,
            player_list=self.player_list,
            register_handle=self.register_handle,
        )
        self.game_index_count = 0
        self.remote_games = {}
        self.add_remote_games(self.game_num)
        # for ga in self.remote_games.values():
        #     data = ga.get_data.remote()

        self.game_last_info = []
        if self.training:
            self.ray_mode = config.ray_mode
            self.data_async = config.data_config.data_async
            if self.config.local_data:
                self.data = [i.get_data.remote() for i in self.remote_games.values()]
            self.episode_data = collections.defaultdict(
                lambda: collections.defaultdict(lambda: [])
            )
        else:
            self.ray_mode = config.eval_config.ray_mode

        if self.ray_mode == "async":
            self.sample_data_by_episode = self._sample_data_by_episode_async
            self.run_episode = self._run_episode_async
        else:
            self.sample_data_by_episode = self._sample_data_by_episode_sync
            self.run_episode = self._run_episode_sync
        self.game_total_steps = 0

    def add_remote_games(self, game_number: int = 1):
        """add remote game

        Args:
            game_number (int, optional): [description]. Defaults to 1.
        """
        for i in range(game_number):
            self.game_index_count += 1
            game_cls = Game.as_remote().remote
            remote_game_i = game_cls(
                config=self.config,
                game_id=self.game_index_count,
                training=self.training,
                env_name=self.env_name,
                player_list=self.player_list,
                register_handle=self.register_handle,
            )
            self.remote_games[self.game_index_count] = remote_game_i

    def sync_weights(
        self,
        train_result=None,
        train_result_ref=None,
        player_w_dict=None,
        player_id=None,
        mode="sync",
    ):
        """sync the model params for agent model in remote game

        Args:
            train_result (dict): dict contains params, train result format
            mode (str): the mode,
                1. sync: make sure all the games has finished the setting of weights before return
                2. async: immediatelly return and return nothing
                3. sync_later: immediatelly return the ray ref used to sync later

        """
        func_returns = []
        if train_result:
            player_id = train_result["player_id"]
            model_id = train_result["model_id"]
            weights = train_result["train_result"]["weights"]
            for game_index, game in self.remote_games.items():
                func_return = game.set_weights.remote(weights, player_id, model_id)
                func_returns.append(func_return)
        elif train_result_ref:
            player_id = train_result_ref["player_id"]
            model_id = train_result_ref["model_id"]
            weights_ref = train_result_ref["train_result"]
            for game_index, game in self.remote_games.items():
                func_return = game.set_ref_weights.remote(
                    weights_ref, player_id, model_id
                )
        elif player_w_dict:
            assert player_id is not None
            for model_id, weights in player_w_dict.items():
                func_returns = []
                for game_index, game in self.remote_games.items():
                    func_return = game.set_weights.remote(weights, player_id, model_id)
                    func_returns.append(func_return)
        else:
            raise ValueError("parmas are all None")
        if mode == "sync":
            result = ray.get(func_returns)
            del result
            del func_returns
            return
        elif mode == "async":
            del func_returns
            return
        elif mode == "sync_later":
            return func_returns
        else:
            raise ValueError("the value of mode must be sync, async or sync_later")

    def sync_player_weights(self, player_id, player_weights):
        """sync the specified player's weights

        Args:
            player_id (str): [description]
            player_weights (Dict): {agent_name:agent_weights}
        """
        for game_index, remote_game in self.remote_games.items():
            remote_game.set_player_weights.remote(player_id, player_weights)

    def get_data_ref(self):
        data_ref_list = [i.get_data.remote() for i in self.remote_games.values()]
        return data_ref_list

    def _sample_data_async(self):
        """get data from remote games on the async mode of ray

        Returns:
            Dict: Game data
        """
        data, self.data = ray.wait(self.data)  # data is list:[real_data]
        data = ray.get(data)[0]
        data_game_id = data.keys()
        for k in data_game_id:
            self.data.append(self.remote_games[k].get_data.remote())
        return data

    def _sample_data_sync(self):
        """get data from remote games on the sync mode of ray

        Returns:
            list: game data list
        """
        data = ray.get(self.data)
        self.data = [i.get_data.remote() for i in self.remote_games.values()]
        return data

    def _sample_data_by_episode_async(self):
        """get episode data, ray async mode

        #TODO: 合并重复的代码

        Returns:
            list: [episode_data,episode_data,...]
        """
        data_to_return = []
        while True:
            data = self._sample_data_async()
            assert len(data) == 1
            g_id = list(data.keys())[0]
            d = data[g_id]
            for d_key, d_data in d.items():
                if d_key != "data_num":
                    self.episode_data[g_id][d_key] += d_data
                else:
                    self.game_total_steps += d_data
            while True in self.episode_data[g_id]["done"]:
                episode_data = {}
                episode_data["data_id"] = g_id
                done_index = self.episode_data[g_id]["done"].index(True)
                for k in self.episode_data[g_id].keys():
                    v = self.episode_data[g_id][k]
                    episode_data[k] = v[: done_index + 1]
                    self.episode_data[g_id][k] = v[done_index + 1 :]

                last_info_tmp = episode_data["game_data"][-1]["info"]
                if not "error" in last_info_tmp:
                    data_to_return.append(episode_data)
                    last_step_info = episode_data["game_data"][-1]
                    self.game_last_info.append(last_step_info)
                else:
                    continue
            if data_to_return:
                break
        return data_to_return

    def _sample_data_by_episode_sync(self):
        """get episode data, ray sync mode

        Returns:
            list: [episode_data,episode_data,...]
        """
        data_to_return = []
        while True:
            data_list = self._sample_data_sync()
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
                    last_info_tmp = episode_data["game_data"][-1]["info"]
                    if not "error" in last_info_tmp:
                        data_to_return.append(episode_data)
                        last_step_info = episode_data["game_data"][-1]
                        self.game_last_info.append(last_step_info)
                    else:
                        continue
            if data_to_return:
                break
        return data_to_return

    def get_game_last_info(self):
        """get the game result info

        Returns:
            Dict: game info
        """
        # game last info is use to note game result during training
        ret = self.game_last_info
        self.game_last_info = []
        return ret

    def _run_episode(self):
        """make all remote games to run an episode

        this method is used to run games and get the game result when eval a player
        Returns:
            list: game result list,ray ref object
        """
        result = []
        for game_id, game in self.remote_games.items():
            game_result = game.run_episode.remote()
            result.append(game_result)
        return result

    def _run_episode_async(self):
        """make all remote games to run an episode

        thie method get the game result with ray async mode

        Returns:
            list: game result list
        """
        result = []
        result.extend(self._run_episode())
        game_results = []
        while result:
            re, result = ray.wait(result)
            game_results.extend(ray.get(re))
        return game_results

    def _run_episode_sync(self):
        """make all remote games to run an episode

        this method is used to run games and get the game result when eval a player
        Returns:
            list: game result list
        """
        result = []
        result.extend(self._run_episode())
        game_results = ray.get(result)
        return game_results

    def run_episodes(self, total_episode_num):
        """run all remote games episode by episode

        Args:
            total_episode_num (int): [description]

        Returns:
            list: game result list
        """
        # TODO:完全异步的评估
        run_num = math.ceil(total_episode_num / self.game_num) + 1
        result = []
        for i in range(run_num):
            re = self.run_episode()
            result.extend(re)
        result = result[0:total_episode_num]
        # self.env_close()
        return result

    def env_close(self):
        for game_id, game in self.remote_games.items():
            game.env_close.remote()

    def set_player(self, player):
        """set player for every game

        Args:
            player (BasePlayer): player
        """
        for i in range(self.game_num):
            game_id = i + 1
            self.remote_games[game_id].set_player.remote(player)
        self.local_game.set_player(player)

    def update_player(self):
        return

    def reset_env(self, env_id=None, env=None):
        """change the env in game

        .. Note::
            Select one of two params to use
        Args:
            env_id (str): game can be game_id or BaseEnv object
            env (BaseEnv): env object
        """
        for game_id, game in self.remote_games.items():
            game.reset_env.remote(env_id=env_id, env=env)

    def reset_player(self, player_id_list: list = None, player=None):
        """reset the player in game

        .. Note::
            players should be adaptive to the env. if game env had been reset,
            should check if the players in game adaptive to the new env

        Args:
            player_id_list (List, optional): player id list tells which players to be used in game. Defaults to None.
            player (BasePlayer, optional): player give the player object used in the game. Defaults to None.

        Raises:
            ValueError: [description]
        """
        for game_id, game in self.remote_games.items():
            game.reset_player.remote(player_id_list=player_id_list, player=player)

    def game_pause(self):
        pass
