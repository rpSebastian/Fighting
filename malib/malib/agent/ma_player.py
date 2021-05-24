from inspect import isfunction
import ray
import torch
from malib.utils import load
from malib.utils import BaseConfig

from .base_player import BasePlayer


class MAPlayer(BasePlayer):
    """BasePlayer object"""

    def __init__(self, player_id, config, training=True, register_handle=None) -> None:
        super(MAPlayer, self).__init__(player_id, config, training, register_handle)
        self.model_dict = self.make_model(self.config)
        self.get_am_mapping()

    def select_action(self, obs):
        """player use this function to generate the action base on the env's observation

        1. compute feature from obs
        2. model forward
        3. generate action

        Args:
            obs (Dict): observation of env

        Returns:
            Action: the action message
        """
        action_mask = self.make_mask(obs)
        fea_dict = self.feature_function(obs)
        fea_use = self.agent_data2model_data(fea_dict)
        model_outs = self.model_forward(fea_use)
        model_outs = self.model_data2agent_data(model_outs)

        player_action, player_data_dict = self.action_function(model_outs, action_mask)
        player_action = self.dict_action2list_action(player_action)
        if self.training:
            p_data = {}
            if "obs" in self.data_to_save:
                p_data["obs"] = obs
            if "feature" in self.data_to_save:
                p_data["feature"] = fea_dict
            if "model_out" in self.data_to_save:
                p_data["model_out"] = model_outs
            for k, v in player_data_dict.items():
                assert k in self.data_to_save
                p_data[k] = v
            assert len(p_data) == len(self.data_to_save)
            # if "action" in self.data_to_save:
            # p_data["action"] = player_action_dict
            # self.data_store_list.append(p_data)
            self.step_data = p_data
        self.total_step_num += 1
        return player_action

    def agent_data2model_data(self, dict_data):
        result = {}
        for m_key, m_agents in self.model2agent.items():
            result[m_key] = []
            for agent_id in m_agents:
                result[m_key].append(dict_data[agent_id])
        return result

    def model_data2agent_data(self, dict_data):
        result = {}
        for m_id, m_data in dict_data.items():
            agent_list = self.model2agent[m_id]
            for i, agent_id in enumerate(agent_list):
                result[agent_id] = m_data[i]
        return result

    def model_forward(self, fea_data):
        model_out = {}
        for m_id, m_in in fea_data.items():
            m_in = torch.tensor(m_in, dtype=torch.float32)
            model_out[m_id] = self.model_dict[m_id](m_in)
        return model_out

    def dict_action2list_action(self, dict_data):
        list_action = []
        for agent_id in self.agent_list:
            list_action.append(dict_data[agent_id])

        return list_action
