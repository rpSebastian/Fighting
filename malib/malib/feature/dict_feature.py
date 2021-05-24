from malib.feature import Feature
import torch


class DictFeature(Feature):
    def __init__(self, config, player_id=None, agent_id=None, training=None):
        super(DictFeature, self).__init__(
            config, player_id=player_id, agent_id=agent_id, training=training
        )

    def player_feature_train(self, env_state):
        f_dict = env_state["obs"]
        return f_dict

    def player_feature_test(self, env_state):
        return self.player_feature_train(env_state)