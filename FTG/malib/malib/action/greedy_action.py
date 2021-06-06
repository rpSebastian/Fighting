import numpy as np
import torch
from malib.action import Action


class GreedyAction(Action):
    def __init__(self, config, player_id=None, agent_id=None, training=None):
        super(GreedyAction, self).__init__(
            config, player_id=player_id, agent_id=agent_id, training=training
        )
        self.player_id = player_id
        self.agent_id = agent_id
        self.training = training
        if self.player_id:
            self.epsilon = config.player_config[self.player_id].action_config.epsilon
            self.episode_count = config.player_config[
                self.player_id
            ].action_config.episode_count
            self.epsilon_enable = config.player_config[self.player_id].action_config.epsilon_enable
        else:
            self.epsilon = config.action_config[agent_id].epsilon
            self.episode_count = config.action_config[agent_id].episode_count
            self.epsilon_enable= config.action_config[agent_id].epsilon_enable

    def agent_action_train(self, model_out):
        pass

    def agent_action_test(self, model_out):
        pass

    def player_action_train(self, model_out: dict, mask=None):
        action_dict = {}
        for agent_id, agent_model_out in model_out.items():
            if np.random.rand() < self.epsilon:
                if mask:
                    avail_actions_ind = np.nonzero(mask[agent_id])[0]
                    action = np.random.choice(avail_actions_ind)
                else:
                    action_dim = agent_model_out.shape[0]
                    action = np.random.randint(0, action_dim)
            else:
                prob = torch.nn.Softmax(dim=-1)(agent_model_out)
                if mask:
                    prob = prob * torch.tensor(mask[agent_id], dtype=torch.float32)
                action_index = torch.argmax(prob)
                action = action_index.item()
            if(self.epsilon_enable==True):
                self.epsilon -= 1.0 / self.episode_count
                self.epsilon = max(0.1, self.epsilon)
            action_dict[agent_id] = action
        player_data_dict = {}
        player_data_dict["action"] = action_dict
        return action_dict, player_data_dict

    def player_action_test(self, model_out: dict, mask=None):
        action_dict = {}
        for agent_id, agent_model_out in model_out.items():
            prob = torch.nn.Softmax(dim=-1)(agent_model_out)
            if mask:
                prob = prob * torch.tensor(mask[agent_id], dtype=torch.float32)
            action_index = torch.argmax(prob)
            # action_index = torch.multinomial(prob, num_samples=1)
            action = action_index.item()
            action_dict[agent_id] = action
        player_data_dict = {}
        player_data_dict["action"] = action_dict
        return action_dict, player_data_dict
