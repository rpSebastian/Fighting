class Action(object):
    def __init__(self, config, player_id=None, agent_id=None, training=None):
        self.training = training
        self.player_id = player_id
        self.agent_id = agent_id
        assert self.training is not None
        if self.training:
            if self.player_id:
                self.gen_action = self.player_action_train
            else:
                self.gen_action = self.agent_action_train
        else:
            if self.player_id:
                self.gen_action = self.player_action_test
            else:
                self.gen_action = self.agent_action_test

    def reset(self):
        pass

    def gen_action(self, model_out):
        pass

    def player_action_train(self, model_out):
        pass

    def player_action_test(self, model_out):
        pass

    def agent_action_train(self, model_out):
        pass

    def agent_action_test(self, model_out):
        pass
