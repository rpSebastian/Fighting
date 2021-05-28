import math
import threading
import time

import ray

from malib.game import VectorGame
from malib.utils import Logger


class Evaluator(object):
    def __init__(self, config, index=0, register_handle=None):
        self.eval_index = index
        self.config = config
        self.register_handle = register_handle
        self.eval_game_num = config.eval_config.eval_game_number
        self.total_episode_number = config.eval_config.total_episode_number
        self.game_episoce_number = math.ceil(
            self.total_episode_number / self.eval_game_num
        )
        # self.game_num = config.eval_config.eval_game_number
        self.eval_games = VectorGame(
            self.config, training=False, register_handle=self.register_handle
        )
        # self.opponent=config.eval_config.opponent
        # self.game_mode = config.eval_config.game_mode

    def set_player_old(self, player):
        """player 应该是：每得到新的模型，保存一个player？还是每得到新
        的模型，只保存模型而不是整个player?应该每个agent异步保存和更新自己的model，还是
        一个player中的所有agent同步保存和更新agent的model
        """

        # TODO:how to set away player
        player_weights = player.get_player_weights()
        player_id = player.player_id
        self.eval_games.sync_player_weights(player_id, player_weights)

    def set_player(self, player):
        self.eval_games.set_player(player)

    def eval_player(self, player, opponent=None):
        """evlaute the player

        use_poopnent to set the away player

        Args:
            player (BasePlayer): player to be evaluated
            opponent (BasePlayer or List): the awayplayer, optional): [description]. Defaults to None.

        Returns:
            Dict: eval result
        """
        self.set_player(player)
        if opponent:
            if type(opponent) is not list:
                self.set_player(opponent)
            else:
                for op in opponent:
                    self.set_player(op)
        eval_result = self.eval_games.run_episodes(self.total_episode_number)
        return eval_result

    def eval_run(self, player, opponent=None, env_id=None):
        """evlaute the player

        use_poopnent to set the away player

        Args:
            player (BasePlayer): player to be evaluated
            opponent (BasePlayer or List): the awayplayer, optional): [description]. Defaults to None.

        Returns:
            Dict: eval result
        """
        player_info = player.info
        opponent_info = None
        self.set_player(player)

        if opponent:
            if type(opponent) is not list:
                self.set_player(opponent)
                opponent_info = opponent.info
            else:
                opponent_info = []
                for op in opponent:
                    opponent_info.append(op.info)
                    self.set_player(op)
        if env_id:
            pass
        eval_result = self.eval_games.run_episodes(self.total_episode_number)
        result = {}
        result["player_info"] = player_info
        result["opponent_info"] = opponent_info
        result["result"] = eval_result
        result["eval_index"] = self.eval_index
        return result

    def reset_env(self, env_id=None, env=None):
        """change the env in game

        .. Note::
            Select one of two params to use
        Args:
            env_id (str): game can be game_id or BaseEnv object
            env (BaseEnv): env object
        """
        self.eval_games.reset_env(env_id=env_id, env=env)

    def reset_player(self, player_id_list=None, player=None):
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
        self.eval_games.reset_player(player_id_list=player_id_list, player=player)

    def set_eval_env(self):
        """set eval away player and environment"""
        pass

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


class VecEvaluator(object):
    def __init__(self, config, standings, register_handle=None):
        self.config = config
        self.register_handle = register_handle
        self.evaluator_num = config.eval_config.evaluator_num
        self.standings = standings
        self.eval_index = 0
        self.evaluators = {}
        self.evaluator_ready = {}
        self.add_evaluators()
        self.player_queue = []
        self.eval_results_ref = []
        self.logger = Logger(config.log_config)
        self.eval_finished = []
        self.eval_runing = []
        self.eval_waiting = []
        self.lock = threading.Lock()

    def start(self):
        self.logger.info("start evaluation ...", stdout=True)
        self.t = threading.Thread(target=self.keep_evaluating)
        self.t.setDaemon(True)
        self.t.start()

    def keep_evaluating(self):
        while True:
            time.sleep(0.4)
            ready_evals = [k for k, v in self.evaluator_ready.items() if v]
            if len(ready_evals) > 0:
                if len(self.player_queue) > 0:
                    player, opponent, env_id = self.player_queue.pop(0)
                    eval_index = ready_evals[0]
                    self.evaluator_ready[eval_index] = False
                    result_ref = self.evaluators[eval_index].eval_run.remote(
                        player, opponent, env_id
                    )
                    self.eval_results_ref.append(result_ref)
                    # evaluator state updata
                    self.lock.acquire()
                    p_name = player.player_name
                    self.eval_waiting.remove(p_name)
                    self.eval_runing.append(p_name)
                    self.lock.release()
            if len(self.eval_results_ref) > 0:
                result_ref, eval_result_ref = ray.wait(self.eval_results_ref, timeout=1)
                assert len(result_ref) <= 1
                if len(result_ref) == 1:
                    self.eval_results_ref = eval_result_ref
                    result = ray.get(result_ref[0])
                    eval_result = result["result"]
                    eval_index = result["eval_index"]
                    self.evaluator_ready[eval_index] = True
                    player_info = result["player_info"]
                    opponent_info = result["opponent_info"]
                    self.standings.update.remote(
                        eval_result, player_info, opponent_info
                    )
                    # evaluator state updata
                    self.lock.acquire()
                    p_name = player_info["player_name"]

                    self.eval_runing.remove(p_name)
                    self.eval_finished.append(p_name)
                    self.lock.release()

    def eval_player(self, player, opponent=None, env_id=None):
        self.lock.acquire()
        self.eval_waiting.append(player.player_name)
        self.lock.release()
        self.player_queue.append((player, opponent, env_id))

    def add_evaluators(self):
        for i in range(self.evaluator_num):
            self.add_evaluator()

    def add_evaluator(self):
        eval_cls = Evaluator.as_remote().remote
        eval_index = self.eval_index
        remote_evaluator = eval_cls(
            config=self.config, index=eval_index, register_handle=self.register_handle
        )
        self.evaluators[eval_index] = remote_evaluator
        self.evaluator_ready[eval_index] = True
        self.eval_index += 1

    def state_info(self):
        self.lock.acquire()
        waiting_str = " ".join(self.eval_waiting)
        finished_str = " ".join(self.eval_finished)
        runing_str = " ".join(self.eval_runing)
        self.lock.release()
        line1 = "waiting player: " + str(len(self.eval_waiting)) + ", " + waiting_str
        line2 = "runing player: " + str(len(self.eval_runing)) + ", " + runing_str
        line3 = "finished player: " + str(len(self.eval_finished)) + ", " + finished_str
        return (line1, line2, line3)

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
