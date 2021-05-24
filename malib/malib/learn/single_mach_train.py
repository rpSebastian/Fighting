import ray
import torch

from malib.utils import load


class SingleMTrain:
    def __init__(
        self, config, model_id=None, register_handle=None
    ) -> None:
        super().__init__()
        self.config = config
        self.register_handle = register_handle
        self.model = None
        self.player_id = config.learn_player_id
        self.model_id = model_id
        self.trainer = None
        self.trainer_mode = config.trainer_config.trainer_mode
        if self.trainer_mode == "local":
            self._train_on_batch = self._train_on_batch_local
        else:
            self._train_on_batch = self._train_on_batch_remote
        self.use_gpu = config.trainer_config.use_gpu
        self.gpu_num = config.trainer_config.gpu_num
        self.add_trainer()

    def add_trainer(self):
        trainer_config = self.config.trainer_config[self.model_id]

        trainer_cls_str = trainer_config.trainer_name
        if type(trainer_cls_str) is str:
            if ":" in trainer_cls_str:
                trainer_cls_str = "malib." + trainer_cls_str
                trainer_cls = load(trainer_cls_str)
            else:
                trainer_cls = ray.get(self.register_handle.get.remote(trainer_cls_str))
        else:
            trainer_cls = trainer_cls_str
        if self.trainer_mode == "remote":
            if self.use_gpu:
                num_gpus = self.gpu_num
            else:
                num_gpus = 0
            trainer_cls_remote = trainer_cls.as_remote(num_gpus=num_gpus).remote
            trainer = trainer_cls_remote(
                self.config,
                player_id=self.player_id,
                model_id=self.model_id,
            )
            if trainer_config.training_procedure:
                training_procedure = trainer_config.training_procedure
                trainer.set_training_procedure.remote(training_procedure)
        else:
            trainer = trainer_cls(
                self.config,
                player_id=self.player_id,
                model_id=self.model_id,
            )
            if trainer_config.training_procedure:
                training_procedure = trainer_config.training_procedure
                trainer.set_training_procedure(training_procedure)
        self.trainer = trainer

    def _train_on_batch_remote(self, data):
        """
        train model on given datas

        Args:
            data (tuple): data to training model
        """
        res = self.trainer.train_on_batch.remote(data)

        return res

    def _train_on_batch_local(self, data):
        """
        train model on given datas

        Args:
            data (tuple): data to training model
        """
        res = self.trainer.train_on_batch(data)

        return res

    def train_on_batch(self, data):
        """[summary]

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        result = self._train_on_batch(data)
        # TODO: 修改返回 list
        return [result]

    def get_weights(self):
        if self.trainer_mode == "remote":
            return ray.get(self.trainer.get_weights.remote())
        else:
            return self.trainer.get_weights()
