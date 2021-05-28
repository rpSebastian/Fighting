from inspect import isfunction
import ray
from malib.utils import load
from malib.utils import BaseConfig

from .base_player import BasePlayer


class SAPlayer(BasePlayer):
    """BasePlayer object"""

    def __init__(self, player_id, config, training=True, register_handle=None) -> None:
        super(SAPlayer, self).__init__(player_id, config, training, register_handle)
        assert len(self.agent_list) == 1
        self.agent_id = self.agent_list[0]
        self.model_dict = self.make_model(self.config)
        self.model_id = list(self.model_dict.keys())[0]
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
        model_outs = {}
        fea_dict = self.feature_function(obs)
        m_out = self.model_dict[self.model_id](fea_dict[self.agent_id])
        model_outs[self.agent_id] = m_out
        player_action, player_data_dict = self.action_function(model_outs)
        player_action = player_action[self.agent_id]
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
