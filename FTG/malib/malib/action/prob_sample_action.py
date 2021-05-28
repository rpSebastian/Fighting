import numpy as np
import torch
from torch.distributions import Categorical
from malib.action import Action


class ProbSampleAction(Action):
    def __init__(self, config, player_id=None, agent_id=None, training=None):
        super(ProbSampleAction, self).__init__(
            config, player_id=player_id, agent_id=agent_id, training=training
        )
        self.player_id = player_id
        self.agent_id = agent_id
        self.training = training

    def agent_action_train(self, model_out):
        pass

    def agent_action_test(self, model_out):
        pass

    def player_action_train(self, model_out: dict, mask=None):
        action_dict = {}
        for agent_id, agent_model_out in model_out.items():
            agent_model_out = torch.nn.Softmax(dim=-1)(agent_model_out)
            if mask:
                agent_model_out = agent_model_out * torch.tensor(
                    mask[agent_id], dtype=torch.float32
                )
            m = Categorical(agent_model_out)
            action = m.sample().item()
            action_dict[agent_id] = action
        player_data_dict = {}
        player_data_dict["action"] = action_dict
        return action_dict, player_data_dict

    def player_action_test(self, model_out: dict, mask=None):
        action_dict = {}
        for agent_id, agent_model_out in model_out.items():
            agent_model_out = torch.nn.Softmax(dim=-1)(agent_model_out)
            if mask:
                agent_model_out = agent_model_out * torch.tensor(
                    mask[agent_id], dtype=torch.float32
                )
            action_index = torch.argmax(agent_model_out)
            action = action_index.item()
            action_dict[agent_id] = action
        player_data_dict = {}
        player_data_dict["action"] = action_dict
        return action_dict, player_data_dict
