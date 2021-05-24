import numpy as np
import torch
from malib.action import Action


class RandomAction(Action):
    def __init__(self, config, player_id=None, agent_id=None, training=None):
        super(RandomAction, self).__init__(
            config, player_id=player_id, agent_id=agent_id, training=training
        )
        self.player_id = player_id
        self.agent_id = agent_id
        self.training = training

    def agent_action_train(self, model_out):
        pass

    def agent_action_test(self, model_out):
        print("hello")
        pass

    def player_action_train(self, model_out: dict):
        action_dict = {}
        action = []
        for agent_id, agent_model_out in model_out.items():
            avail_actions_ind = np.nonzero(agent_model_out)[0]
            ac = np.random.choice(avail_actions_ind)
            action_dict[agent_id] = ac
            action.append(ac)
        player_data_dict = {}
        player_data_dict["action"] = action_dict
        return action, player_data_dict

    def player_action_test(self, model_out: dict):
        return self.player_action_train(model_out)
