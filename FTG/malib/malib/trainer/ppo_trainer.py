from copy import deepcopy

import torch

from malib.trainer import Trainer


class PPOTrainer(Trainer):
    def __init__(
        self,
        config,
        player_id,
        model_id=None,
        trainer_index=0,
        register_handle=None,
    ):
        super(PPOTrainer, self).__init__(
            config,
            player_id,
            trainer_index,
            model_id=model_id,
            register_handle=register_handle,
        )
        self.loss_func = torch.nn.MSELoss()
        self.iteration_count = 0
        self.gamma = config.trainer_config[self.model_id].gamma
        self.lmbda = config.trainer_config[self.model_id].lmbda
        self.es_clip = config.trainer_config[self.model_id].es_clip
        self.loss_func = torch.nn.SmoothL1Loss()
        self.epoch = config.trainer_config[self.model_id].epoch

    def _train_on_batch(self, data):
        # self.update_target_model()
        # self.target_model_update_iter+=1
        # print("DQN train on batch")
        feature_list = data.feature
        fea0 = [f[0] for f in feature_list]
        fea1 = [f[1] for f in feature_list]
        action = [a[0] for a in data.action]
        model_out = [mo[0] for mo in data.model_out]
        reward = [gd[0]["reward"] for gd in data.game_data]
        done = [gd[0]["done"] for gd in data.game_data]
        dataset = data.make_dataset([fea0, fea1, action, done, reward, model_out])
        loss_info = []
        for e in range(self.epoch):
            for d in dataset:
                f0, f1, ac, do, re, mo = d
                f0, f1, ac, do, re, mo = (
                    f0.to(self.device),
                    f1.to(self.device),
                    ac.to(self.device),
                    do.to(self.device),
                    re.to(self.device),
                    mo.to(self.device),
                )
                re = re.float().view(-1, 1) / 100
                ac = ac.view(-1, 1)
                do = do.view(-1, 1)
                done_mask = do == 0
                td_target = re + self.gamma * self.model.v(f1) * done_mask
                delta = td_target - self.model.v(f0)
                delta = delta.detach()
                adv_list = []
                adv = 0.0
                for i in range(delta.shape[0] - 1, -1, -1):
                    delta_i = delta[i]
                    adv = self.gamma * self.lmbda * adv + delta_i[0]
                    adv_list.append([adv])
                adv_list.reverse()
                adv = torch.tensor(adv_list, dtype=torch.float)

                p_out = self.model(f0)
                p_out = p_out.gather(1, ac)
                ratio = torch.exp(torch.log(p_out) - torch.log(mo))
                adv = adv.to(self.device)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.es_clip, 1 + self.es_clip) * adv
                loss = -torch.min(surr1, surr2) + self.loss_func(
                    self.model.v(f0), td_target.detach()
                )

                loss = loss.mean()
                loss_info.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.iteration_count += 1

        result = {}
        result["loss"] = loss_info
        return result