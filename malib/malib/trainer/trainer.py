import time
import types
from abc import ABC, abstractmethod

import ray
import torch
import torch.nn as nn
import numpy as np
from malib.utils import make_model
from malib.model import DataParallel
from malib.utils import Logger, load
from malib.data import Data


class Trainer(ABC):
    def __init__(
        self,
        config,
        player_id,
        trainer_index,
        model_id=None,
        register_handle=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.register_handle = register_handle
        self.local_data = config.local_data
        if self.config.trainer_config.use_gpu:
            self.use_gpu = True
            self.gpu_num = config.trainer_config.gpu_num
            self.device = torch.device("cuda:0")
        else:
            self.use_gpu = False
            self.device = torch.device("cpu")
        self.player_id = player_id
        self.model_id = model_id
        self.trainer_index = 0
        self.model = self.make_model(
            self.model_id,
        )
        self.model.to(self.device)
        if self.gpu_num > 1:
            self.model = DataParallel(self.model)
        self.training_result = {}
        self.training_result["model_id"] = self.model_id
        self.training_result["trainer_index"] = trainer_index
        self.train_count = 0
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.config.trainer_config[self.model_id].lr
        )
        self.iteration_count = 0
        self.logger = Logger(self.config.log_config)

        self.current_loss_info = None

    @abstractmethod
    def _train_on_batch(self, data):
        return None

    def train_on_batch(self, data):
        # print("Trainer train on batch")
        if not self.local_data:
            agent_data_ref = list(data.values())
            agent_data = ray.get(agent_data_ref)
        else:
            agent_data = list(data.values())
        data = agent_data.pop(0)
        data.extend(agent_data)
        train_info = {}
        result = self._train_on_batch(data)
        self.current_loss_info = result["loss"]
        loss = np.mean(result["loss"])
        self.logger.add_scalar(self.model_id + "/training_loss", loss, self.train_count)

        result["weights"] = self.get_model_weights()
        train_info["iter_num"] = self.iteration_count
        train_info["player_id"] = self.player_id
        train_info["model_id"] = self.model_id
        train_info["train_result"] = result
        self.train_count += 1
        return train_info

    def set_training_procedure(self, training_procedure):
        self._train_on_batch = types.MethodType(training_procedure, self)

    def get_weights(self):
        weights = self.get_model_weights()
        result = {}
        result["weights"] = weights
        train_info = {}
        train_info["player_id"] = self.player_id
        train_info["model_id"] = self.model_id
        train_info["train_result"] = result
        return train_info

    def get_model_weights(self):
        weights = self.model.get_weights()
        if self.use_gpu:
            weights = {k: v.cpu().detach() for k, v in weights.items()}
        return weights

    def make_model(self, model_id=None):
        if self.model_id:
            model_name = self.config.player_config.model_config[model_id].model_name
            model_params = self.config.player_config.model_config[model_id].model_params
            model_params["device"] = self.device
            if type(model_name) is str:
                if ":" in model_name:
                    model_name = "malib." + model_name
                    cls = load(model_name)
                else:
                    assert self.register_handle is not None
                    cls = ray.get(self.register_handle.get.remote(model_name))
            else:
                cls = model_name
            model = cls(**model_params)
        else:
            raise ValueError
        return model

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
