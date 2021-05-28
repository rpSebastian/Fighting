import collections
import copy
import threading
import time

import numpy as np
import ray

from malib.data import DataStorage
from malib.game import VectorGame
from malib.learn import Evaluator, SingleMTrain
from malib.utils import Logger, load


class Learner(object):
    def __init__(self, config, learning_procedure=None, register_handle=None):
        # init learner with config file
        self.config = config
        self.register_handle = register_handle
        self.player_id = config.player_id
        self.getting_data = False
        self.player = self.make_player()
        self.agent2model = self.player.agent2model
        self.model2agent = self.player.model2agent
        self.agent_list = config.player_config[self.player_id].agents

        # self.learn_agent_id_list = None
        self.learn_model_id_list = config.learn_model_id
        self.local_data = config.local_data
        self.build_data_storage()
        # learner_pro=learning_procedure(self)
        # self.learning_procedure = types.MethodType(learner_pro,self)
        self.logger = Logger(config.log_config)
        self.learn_step_number = 0
        self.trainer_mode = config.trainer_config.trainer_mode
        self.ray_mode = config.ray_mode
        # if self.local_data:
        if True:
            if self.trainer_mode == "local":
                self.ray_get_later = False
                self.learn_on_data = self._learn_on_data_local
            else:
                self.ray_get_later = True
                if self.ray_mode == "async":
                    self.learn_on_data = self._learn_on_data_remote
                else:
                    self.learn_on_data = self._learn_on_data_remote_sync

        # else:
        #     if self.trainer_mode == "local":
        #         self.ray_get_later = False
        #         self.learn_on_data = self._learn_on_data_local
        #     else:
        #         self.ray_get_later = True
        #         if self.ray_mode == "async":
        #             self.learn_on_data = self._learn_on_data_remote
        #         else:
        #             self.learn_on_data = self._learn_on_data_remote_sync

        if self.local_data:
            self.ask_for_data = self._ask_for_data_sync
        else:
            self.ask_for_data = self._ask_for_data_ref_sync
        if learning_procedure:
            self.learning_procedure = learning_procedure(self)
        self.train_iter_num = collections.defaultdict(lambda: 0)
        self.game_total_steps = 0

        self.lock = threading.Lock()

    def run(self):
        """
        run the algorithm
        """
        while True:
            result = self.step()

    def step(self):
        """run one step of learning"""
        self._step()
        self.learn_step_number += 1
        self.player.update_player_learn_step_num(self.learn_step_number)

    def _step(self):
        """
        run algorithm step by step

        """
        l_step_result = self.learning_procedure(self)
        # self.logger.add_dict(l_step_result)

        return l_step_result

    def _learn_on_data_remote(self, data):
        learn_result_ref = []
        for model_id in self.learn_model_id_list:
            model_data = self.union_data(data, model_id)
            train_result_ref = self.trainers[model_id].train_on_batch(model_data)
            if self.ray_get_later:
                ref_dict = {}
                ref_dict["player_id"] = self.player_id
                ref_dict["model_id"] = model_id
                ref_dict["train_result"] = train_result_ref
                learn_result_ref.append(ref_dict)
            else:
                learn_result_ref.extend(train_result_ref)
        if self.ray_get_later:
            return learn_result_ref
        else:

            learn_result = []
            while learn_result_ref:
                train_result, learn_result_ref = ray.wait(learn_result_ref)
                learn_result.extend(ray.get(train_result))

            return learn_result

    def get_game_info(self):
        if self.local_data:
            game_info = self.vector_game.get_game_last_info()
            return game_info
        else:
            game_info = self.data_storage.get_game_last_info.remote()
            game_info = ray.get(game_info)
            return game_info

    def clear_data(self):
        if self.local_data:
            self.data_storage.clear_data()
        else:
            self.data_storage.clear_data.remote()

    def union_data(self, data, model_id):
        agent_list = self.model2agent[model_id]
        new_data = {}
        for agent_id in agent_list:
            new_data[agent_id] = data[agent_id]
        return new_data

    def _learn_on_data_remote_sync(self, data):
        learn_result_ref = []
        for model_id in self.learn_model_id_list:
            model_data = self.union_data(data, model_id)
            train_result_ref = self.trainers[model_id].train_on_batch(model_data)
            if self.ray_get_later:
                ref_dict = {}
                ref_dict["player_id"] = self.player_id
                ref_dict["model_id"] = model_id
                ref_dict["train_result"] = train_result_ref
                learn_result_ref.append(ref_dict)
            else:
                learn_result_ref.extend(train_result_ref)

        if self.ray_get_later:
            return learn_result_ref
        else:

            learn_result = ray.get(learn_result_ref)

            # tboard_info = self.create_tboard_info(learn_result)
            # self.logger.add_dict(tboard_info)

            return learn_result

    def _learn_on_data_local(self, data):
        learn_result = []
        for model_id in self.learn_model_id_list:
            model_data = self.union_data(data, model_id)
            train_result_ref = self.trainers[model_id].train_on_batch(model_data)
            learn_result.extend(train_result_ref)
        # tboard_info = self.create_tboard_info(learn_result)
        # self.logger.add_dict(tboard_info)

        return learn_result

    def _learn_on_ref_data_local(self, data):
        pass

    def update_game_total_steps(self):
        self.game_total_steps = self.vector_game.game_total_steps

    def create_tboard_info(self, learn_result):
        train_log = {}
        for lres in learn_result:
            model_id = lres["model_id"]
            loss_list = lres["train_result"]["loss"]
            loss = np.mean(loss_list)
            self.train_iter_num[model_id] += lres["iter_num"]
            log_name = "{}/{}".format(model_id, "loss")
            train_log[log_name] = [loss, self.learn_step_number]

        return train_log

    def sync_weights(
        self,
        train_result=None,
        player_w_dict=None,
        player_id=None,
        mode="sync",
        init_sync=False,
    ):
        """Synchronize the local player and remote player weights from train_result

        .. Note::
            be careful to the params. two ways to set the weights
            1. set weights from trainer, use the trian_result parameter
            2. set weights from player_weights, use the player_w_dict and player_id params

        Args:
            train_result (dict): train result,eg, {'agent_id':xx, 'player_id':xx,'train_result':xx }
            player_w_dict (dict): player weights dict. eg, {'agent_id':xx}
            player_id (str)
            mode (str): the mode,
                1. sync: make sure all the games in vectorgame have finished the setting
                of weights before return
                2. async: immediatelly return and return nothing
                3. sync_later: immediatelly return the ray ref used to sync later

        """
        ray_ref = None
        if train_result:
            if type(train_result) is list:
                for i in train_result:

                    self.sync_weights(train_result=i)
            else:
                if self.ray_get_later and not init_sync:
                    ray_ref = self.vector_game.sync_weights(
                        train_result_ref=train_result, mode=mode
                    )
                    model_id = train_result["model_id"]
                    weights = train_result["train_result"]
                    self.player.set_ref_weights(weights, model_id, later=True)
                else:
                    ray_ref = self.vector_game.sync_weights(
                        train_result=train_result, mode=mode
                    )

                    model_id = train_result["model_id"]
                    weights = train_result["train_result"]["weights"]
                    self.player.set_weights(weights, model_id)
        elif player_w_dict:
            ray_ref = self.vector_game.sync_weights(
                player_w_dict=player_w_dict, player_id=player_id, mode=mode
            )
            assert player_id is not None
            if player_id == self.player.player_id:
                for model_id, weights in player_w_dict.items():
                    self.player.set_weights(weights, model_id)
        else:
            raise ValueError("params are all None")

        if mode == "sync_later":
            assert ray_ref is not None
            return ray_ref
        elif mode == "async" or mode == "sync":
            pass
        else:
            raise ValueError("mode value must be sync_later, sync or async")

    def sync_player_weights(self, player_id, player_weights):
        """set player weights for every game in vector_game

        Args:
            player_id (str): player id
            player_weights (dict): weights dict return from player.get_weights(),
                {'a0':weights_dict,'a1':weights_dict}
        """
        self.vector_game.sync_player_weights(player_id, player_weights)

    # def sync_weights2(self, train_result):
    #     self.eval_game.sync_weights(train_result)

    def init_games_weights(self, player_w_dict=None, player_id=None, mode="sync"):
        """init all the weights of players in all games

        .. NOTE::
            be careful to the params,
            1. player_w_dict is None: init all the weights from trainer
            2. player_w_dict is not None, player_id must be not None, then init all the weights
            from player_w_dict


        Args:
            player_w_dict (dict):
            player_id (str):
            mode (str): sync or async
                sync: make sure all the games have finished setting weights before return
                async: immediatelly return
        """
        if mode == "sync":
            ray_refs = []
            if player_w_dict:
                ray_ref = self.sync_weights(
                    player_w_dict=player_w_dict,
                    player_id=player_id,
                    mode="sync_later",
                    init_sync=True,
                )
                ray_refs.extend(ray_ref)

            else:
                for trainer in self.trainers.values():
                    ray_ref = self.sync_weights(
                        trainer.get_weights(), mode="sync_later", init_sync=True
                    )
                    ray_refs.extend(ray_ref)
            result = ray.get(ray_refs)
            del result
            del ray_refs
        elif mode == "async":
            if player_w_dict:
                assert player_id is not None
                self.sync_weights(
                    player_w_dict=player_w_dict,
                    player_id=player_id,
                    mode="async",
                    init_sync=True,
                )
            else:
                for trainer in self.trainers.values():
                    self.sync_weights(
                        trainer.get_weights(), mode="async", init_sync=True
                    )
        else:
            raise ValueError("value of mode must be sync or async")

    def get_data(self):
        return self.vector_game.sample_data()

    def get_data_by_episode(self):
        data = self.vector_game.sample_data_by_episode()
        self.update_game_total_steps()
        return data

    def build_data_storage(self):
        if self.local_data:
            self.data_storage = DataStorage(
                self.config,
                agent_list=self.agent_list,
                register_handle=self.register_handle,
            )
        else:
            data_storage_cls = DataStorage.as_remote().remote
            self.data_storage = data_storage_cls(
                self.config,
                agent_list=self.agent_list,
                register_handle=self.register_handle,
            )

    def build_games(self):

        print("build_games")
        self.vector_game = VectorGame(self.config, register_handle=self.register_handle)

    # def build_eval_games(self):
    #     self.eval_game = VectorGame(self.config, training=False)

    def build_trainers(self):
        print("build_trainers")
        self.trainers = {}
        for model_id in self.learn_model_id_list:
            trainer = SingleMTrain(
                self.config, model_id=model_id, register_handle=self.register_handle
            )
            self.trainers[model_id] = trainer

    def build_evaluater(self):
        print("build_evaluatere")
        self.evaluator = Evaluator(self.config, register_handle=self.register_handle)

    # TODO: move make players to utils to be a public method
    def make_player(self):
        player_cls_str = self.config.player_config[self.player_id].player_name
        if type(player_cls_str) is str:
            if ":" in player_cls_str:
                player_cls_str = "malib." + player_cls_str
                cls = load(player_cls_str)
            else:
                cls = ray.get(self.register_handle.get.remote(player_cls_str))
        else:
            cls = player_cls_str
        player = cls(
            self.player_id,
            self.config,
            training=False,
            register_handle=self.register_handle,
        )
        return player

    def set_player(self, player):
        """set the player of all games

        .. NOTE::
            this method is only using for resetting the player which is
            not for data generation

        Args:
            player (BasePlayer): player
        """
        player.training = True
        self.vector_game.set_player(player)

    def start_data_thread(self):
        # TODO: add stop and pause method
        self.getting_data = True
        if self.local_data:
            self.ask_for_data = self._ask_for_data_async
            self.t = threading.Thread(target=self.get_data_in_thread)
        else:
            self.ask_for_data = self._ask_for_data_ref_async
            self.t = threading.Thread(target=self.get_data_ref_in_thread)
        self.t.setDaemon(True)
        self.t.start()

    def get_data_in_thread(self):
        while self.getting_data:
            episode_data = self.get_data_by_episode()
            self.lock.acquire()
            self.data_storage.add_episode_data(episode_data)
            self.lock.release()
            time.sleep(0.001)

    def get_data_ref_in_thread(self):
        while self.getting_data:
            data_ref_list = self.vector_game.get_data_ref.remote()
            self.data_storage.add_data_ref.remote(data_ref_list)
            time.sleep(0.001)

    def _ask_for_data_async(self, min_episode_count=0, ask_interval=0.06):
        """learner get data from data_storage, waiting data_storage to prepare
         data, if ready,then return data

        Args:
            min_episode_count (int, optional): [description]. Defaults to 0.
            ask_interval (float, optional): [description]. the time interval of
             visiting to data_storage,if too small, data_storage will be stressful,
             Defaults to 0.1.
        """
        while True:
            if (
                self.data_storage.ready
                and len(self.vector_game.game_last_info) > min_episode_count
            ):
                break
            else:
                time.sleep(ask_interval)
        self.lock.acquire()
        data = self.data_storage.get_data()
        self.lock.release()
        return data

    def _ask_for_data_sync(self, min_episode_count=0, ask_interval=None):
        while True:
            if (
                self.data_storage.ready
                and len(self.vector_game.game_last_info) > min_episode_count
            ):
                break
            else:
                episode_data = self.get_data_by_episode()
                self.data_storage.add_episode_data(episode_data)

        data = self.data_storage.get_data()
        return data

    def _ask_for_data_ref_sync(self, min_episode_count=0, ask_interval=None):
        while True:
            ready_ref = self.data_storage.data_ready.remote()
            data_ref_list = self.vector_game.get_data_ref()
            self.data_storage.add_data_ref.remote(data_ref_list)
            ready = ray.get(ready_ref)
            if ready:
                break
        data_ref = self.data_storage.get_data_ref.remote()
        data_ref_dict = ray.get(data_ref)
        return data_ref_dict

    def _ask_for_data_ref_async(self, min_episode_count=0, ask_interval=0.01):
        # TODO:是否需要加线程锁？
        ready_ref = self.data_storage.data_ready.remote()
        while True:
            ready = ray.get(ready_ref)
            if ready:
                break
            ready_ref = self.data_storage.data_ready.remote()
            time.sleep(ask_interval)
        data_ref = self.data_storage.get_data_ref.remote()
        data_ref_dict = ray.get(data_ref)
        return data_ref_dict

    def get_training_player(self):
        """get the copy of the training player

        Returns:
            BasePlayer: the copy of training player
        """
        return copy.deepcopy(self.player)

    def get_training_weights(self):
        """return the player weights dict of the training player

        Returns:
            dict: player weights dict,ie. {'a0':weights_dict,'a1':weights_dict}
        """
        return self.player.get_player_weights()

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
