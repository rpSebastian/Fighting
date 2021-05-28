import ray
import torch

from malib.utils import load


class VectorTrain:
    def __init__(self, config, agent_id) -> None:
        super().__init__()
        self.config = config
        self.model = None
        self.trainer_num = config.trainer_number
        self.player_id = config.learn_player_id
        self.agent_id = agent_id
        self.remote_trainers = {}
        self.add_remote_trainers()

    def add_remote_trainers(self):
        trainer_config = self.config.trainer_config[self.agent_id]

        trainer_cls_str = trainer_config.trainer_name
        trainer_cls_str = "malib." + trainer_cls_str
        for i in range(self.trainer_num):
            trainer_cls = load(trainer_cls_str)
            trainer_cls_remote = trainer_cls.as_remote().remote
            remote_trainer_i = trainer_cls_remote(
                self.config, agent_id=self.agent_id, trainer_index=i
            )
            if hasattr(trainer_config, "training_procedure"):
                training_procedure = trainer_config.training_procedure
                remote_trainer_i.set_training_procedure.remote(training_procedure)
            self.remote_trainers[i] = remote_trainer_i

    def _train_on_batch(self, data):
        """
        train model on given datas

        Args:
            data (tuple): data to training model
        """
        ray_inf_list = []
        for k, v in self.remote_trainers.items():
            res = v.train_on_batch.remote(data)
            ray_inf_list.append(res)
        # result = ray.get(ray_inf_list)

        return ray_inf_list

    def train_on_batch(self, data):
        return self._train_on_batch(data)
