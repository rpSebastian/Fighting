class Feature(object):
    def __init__(self, config, player_id=None, agent_id=None, training=None):
        self.config = config
        self.training = training
        self.player_id = player_id
        self.agent_id = agent_id
        assert self.training is not None
        if self.training:
            if player_id:
                self.cpt_feature = self.player_feature_train
            else:
                self.cpt_feature = self.agent_feature_train
        else:
            if self.player_id:
                self.cpt_feature = self.player_feature_test
            else:
                self.cpt_feature = self.agent_feature_test

    def cpt_feature(self, env_state):
        pass

    def player_feature_train(self, env_state):
        pass

    def player_feature_test(self, env_state):
        pass

    def agent_feature_train(self, env_state):
        pass

    def agent_feature_test(self, env_state):
        pass
