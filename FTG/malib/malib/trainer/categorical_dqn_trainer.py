from copy import deepcopy

import torch
import numpy as np
import torch.nn.functional as F
from malib.trainer import Trainer


class CategoricalDQNTrainer(Trainer):
    def __init__(
        self, config, player_id, model_id=None, trainer_index=0, register_handle=None
    ):
        super(CategoricalDQNTrainer, self).__init__(
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
        self.noisy = self.config.player_config.model_config[self.model_id].model_params["noisy"]
        self.v_max = self.config.player_config.model_config[self.model_id].model_params["v_max"]
        self.v_min = self.config.player_config.model_config[self.model_id].model_params["v_min"]
        self.atom_size = self.config.player_config.model_config[self.model_id].model_params["atom_size"]
        self.support = torch.cat([
            torch.linspace(self.v_min, self.v_min / 2 - 0.2, self.atom_size // 8), 
            torch.linspace(self.v_min / 2 + 0.2, self.v_max / 2 - 0.2, self.atom_size // 8 * 6), 
            torch.linspace(self.v_max / 2 + 0.2, self.v_max, self.atom_size // 8)
        ]).to(self.device)
        self.n_steps = config.data_config.tra_len
        self.double = config.trainer_config[self.model_id].double
        self.batch_size = self.config.data_config.batch_size
        self.iteration_count = 0
    
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
            re = re.view(-1, 1).float()
            ac = ac.view(-1, 1)
            do = do.view(-1, 1).float()

            # Categorical DQN algorithm
            delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

            with torch.no_grad():
                if self.double:
                    next_action = self.model(f1).argmax(1) # [B]
                else:
                    next_action = self.target_model(f1).argmax(1) # [B]
                next_dist = self.target_model.dist(f1) # [B, A, D]
                next_dist = next_dist[range(self.batch_size), next_action] # [B, D]

                t_z = re + (1 - do) * np.power(self.GAMMA, self.n_steps) * self.support
                t_z = t_z.clamp(min=self.v_min, max=self.v_max)
                b = (t_z - self.v_min) / delta_z
                l = b.floor().long()
                u = b.ceil().long()

                offset = (
                    torch.linspace(
                        0, (self.batch_size - 1) * self.atom_size, self.batch_size
                    ).long()
                    .unsqueeze(1)
                    .expand(self.batch_size, self.atom_size)
                    .to(self.device)
                )

                proj_dist = torch.zeros(next_dist.size(), device=self.device)
                proj_dist.view(-1).index_add_(
                    0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
                )
                proj_dist.view(-1).index_add_(
                    0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
                )

            dist = self.model.dist(f0)
            action = ac.squeeze()
            log_p = torch.log(dist[range(self.batch_size), action])
            loss = -(proj_dist * log_p).sum(1).mean()

            loss = loss.float()
            loss_info.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.noisy:
                # NoisyNet: reset noise
                self.model.reset_noise()
                self.target_model.reset_noise()
            
            self.iteration_count += 1
            if self.iteration_count % self.target_model_update_iter == 0:
                self.update_target_model()

        result = {}
        result["loss"] = loss_info
        return result

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
