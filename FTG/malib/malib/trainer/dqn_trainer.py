from copy import deepcopy

import torch
import numpy as np
import torch.nn.functional as F
from malib.trainer import Trainer


class DQNTrainer(Trainer):
    def __init__(
        self, config, player_id, model_id=None, trainer_index=0, register_handle=None
    ):
        super(DQNTrainer, self).__init__(
            config,
            player_id,
            trainer_index,
            model_id=model_id,
            register_handle=register_handle,
        )
        self.target_model = deepcopy(self.model)
        self.target_model.to(self.device)
        
        loss_func_dict = {
            "MSELoss": torch.nn.MSELoss(),
            "smooth_l1_loss": F.smooth_l1_loss
        }
        self.loss_func = loss_func_dict[self.config.trainer_config.loss_func]
        self.target_model_update_iter = self.config.trainer_config[
            self.model_id
        ].target_model_update_iter
        self.EPSILON = self.config.trainer_config[self.model_id].EPSILON
        self.GAMMA = self.config.trainer_config[self.model_id].GAMMA
        self.iteration_count = 0
        self.n_steps = config.data_config.tra_len
        self.double = config.trainer_config[self.model_id].double

    def _calc_reward(self, tra_rewards):
        rewards = [tra_rewards[i] for i in range(self.n_steps)]
        n_step_return = 0
        for r in reversed(rewards):
            n_step_return = n_step_return * self.GAMMA + r
        return n_step_return

    def _train_on_batch(self, data):
        # self.update_target_model()
        # self.target_model_update_iter+=1
        # print("DQN train on batch")
        feature_list = data.feature
        fea0 = [f[0] for f in feature_list]
        fea1 = [f[self.n_steps] for f in feature_list]
        action = [a[0] for a in data.action]
        reward = [self._calc_reward(r) for r in data.reward] 
        done = [gd[self.n_steps - 1]["done"] for gd in data.game_data]
        dataset = data.make_dataset([fea0, fea1, action, done, reward])
        loss_info = []
        for d in dataset:
            f0, f1, ac, do, re = d
            f0, f1, ac, do, re = (
                f0.to(self.device),
                f1.to(self.device),
                ac.to(self.device),
                do.to(self.device),
                re.to(self.device),
            )
            re = re.float()
            # f0,f1,do,re=f0.float(),f1.float(),do.float(),re.float()
            ac = ac.view(-1, 1)
            q_predict = self.model(f0)
            q_predict = q_predict.gather(1, ac)
            if self.double:
                max_q = self.target_model(f1).gather(
                    1, self.model(f1).argmax(dim=1, keepdim=True)
                ).detach().squeeze()
            else:
                q_next = self.target_model(f1).detach()
                max_q = q_next.max(1)[0]
            n_d = do == 0
            q_next = max_q * n_d
            q_target = re + np.power(self.GAMMA, self.n_steps) * q_next
            q_target = q_target.view(-1, 1)
            loss = self.loss_func(q_predict, q_target)
            loss = loss.float()
            loss_info.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.iteration_count += 1
            if self.iteration_count % self.target_model_update_iter == 0:
                self.update_target_model()

        result = {}
        result["loss"] = loss_info
        return result

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
