import ray
from malib.utils import load


class BaseAgent(object):
    def __init__(
        self,
        agent_id,
        config,
        feature_func=None,
        action_func=None,
        training=True,
        register_handle=None,
    ) -> None:
        super().__init__()
        self.agent_id = agent_id
        self.config = config
        self.register_handle = register_handle
        self.feature_func = feature_func
        self.action_funnc = action_func
        self.training = training
        self.data_store = []
        self.total_step_num = 0

    def reset(self):
        """reset the agent state"""
        pass

    def compute(self, feature):
        """use model to compute action prob

        Args:
            feature (tensor tuple): [description]

        Returns:
            tensor tuple: action probs
        """
        # if self.feature_func:
        #     feature=self.feature_func(feature)
        model_out = self.model(feature)
        # action=self.action_funnc(model_out)
        # if self.training:
        #     # save the agent's data
        #     self.data_store.append([model_out])
        return model_out

    def get_data(self):
        """get the agent data generated when playing game

        Returns:
            [type]: [description]
        """
        data = self.data_store
        self.data_store = []
        return data

    def set_weights(self, weights):
        """set the weights of agent model

        Args:
            weights (dict): model weight dict
        """
        self.model.set_weights(weights)

    def get_weights(self):
        """return the model weights of agent

        Returns:
            dict: pytorch model weights dict
        """
        return self.model.get_weights()

    # TODO: use make_model func in util.py
    def make_model(self, config):
        model_name = config.agent_config[self.agent_id].model_name
        input_shape = config.agent_config[self.agent_id].model_in_shape
        output_shape = config.agent_config[self.agent_id].model_out_shape
        if type(model_name) is str:
            if ":" in model_name:
                model_name = "malib." + model_name
                cls = load(model_name)
            else:
                assert self.register_handle is not None
                cls = ray.get(self.register_handle.get.remote(model_name))
        else:
            cls = model_name
        self.model = cls(in_dim=input_shape, out_dim=output_shape)

    # TODO: make this method to a public api function
    # def load(self, name):
    #     import importlib

    #     mod_name, attr_name = name.split(":")
    #     mod = importlib.import_module(mod_name)
    #     fn = getattr(mod, attr_name)
    #     return fn
