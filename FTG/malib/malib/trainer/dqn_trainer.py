from copy import deepcopy

import torch

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
        self.loss_func = torch.nn.MSELoss()
        self.target_model_update_iter = self.config.trainer_config[
            self.model_id
        ].target_model_update_iter
        self.EPSILON = self.config.trainer_config[self.model_id].EPSILON
        self.GAMMA = self.config.trainer_config[self.model_id].GAMMA
        self.TYPE = self.config.trainer_config[self.model_id].TYPE
        self.iteration_count = 0

    def _train_on_batch(self, data):
        # self.update_target_model()
        # self.target_model_update_iter+=1
        # print("DQN train on batch")
        feature_list = data.feature
        fea0 = [f[0] for f in feature_list]
        fea1 = [f[1] for f in feature_list]
        action = [a[0] for a in data.action]
        reward = [gd[0]["reward"] for gd in data.game_data]
        done = [gd[0]["done"] for gd in data.game_data]
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

            if (self.TYPE.find('DOUBLE')!=-1):
                q_next = self.model(f1).detach() #DDQN
            else:
                q_next = self.target_model(f1).detach() #DQN

            max_q = q_next.max(1)[0]
            n_d = do == 0
            q_next = max_q * n_d.float()
            q_target = re + self.GAMMA * q_next
            q_target = q_target.view(-1, 1)
            loss = self.loss_func(q_predict, q_target)
            loss = loss.float()
            loss_info.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (self.TYPE.find('NOISY')!=-1):
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
