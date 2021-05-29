from inspect import isfunction
import ray
from malib.utils import load
from malib.utils import BaseConfig
import collections


class BasePlayer(object):
    """BasePlayer object"""

    def __init__(self, player_id, config, training=True, register_handle=None) -> None:
        super().__init__()
        self.training = training
        self.register_handle = register_handle
        if self.training:
            self.data_to_save = config.data_config.data_to_save.player_data
        self.player_id = player_id
        self.agent_list = config.player_config[self.player_id].agents
        self.ref_weights = {}
        self.weights = None  # 单个模型或者多个模型的参数
        self.player_name = None  # used in league
        self.player_long_name = None  # used in league
        self.learn_step_number = None  # used in league
        self.agent_dict = None
        self.config = config
        self.action_function = self.setup_action()
        self.feature_function = self.setup_feature()
        # self.agent_dict = self.build_agent(config)
        self.reset_step_data()
        self.total_step_num = 0
        self.train_timestamp = 0
        self.info = {
            "player_id": self.player_id,
            "player_name": self.player_name,
            "player_long_name": self.player_long_name,
            "learn_step_number": self.learn_step_number,
        }

    def update_timestamp(self, timestamp):
        self.train_timestamp = timestamp

    def setup_action(self):
        action_param = self.config.player_config[self.player_id].action_config
        if type(action_param) is str:
            assert self.register_handle is not None
            action_obj = ray.get(self.register_handle.get.remote(action_param))
            if isfunction(action_obj):
                return action_obj
            else:
                action_cls = action_obj
                self.action = action_cls(
                    self.config, player_id=self.player_id, training=self.training
                )

                def gen_action(model_out, action_mask=None):
                    return self.action.gen_action(model_out, action_mask)

                return gen_action

        elif type(action_param) is BaseConfig:
            action_name = action_param.action_name
            action_obj = ray.get(self.register_handle.get.remote(action_name))
            if not isfunction(action_obj):
                self.action = action_obj(
                    self.config, player_id=self.player_id, training=self.training
                )

                def gen_action(model_out, action_mask=None):
                    return self.action.gen_action(model_out, action_mask)

                return gen_action

        else:
            return action_param

    def setup_feature(self):
        feature_param = self.config.player_config[self.player_id].feature_config
        if type(feature_param) is str:
            assert self.register_handle is not None
            feature_obj = ray.get(self.register_handle.get.remote(feature_param))
            if isfunction(feature_obj):
                return feature_obj
            else:
                feature_cls = feature_obj
                self.feature = feature_cls(
                    self.config, player_id=self.player_id, training=self.training
                )

                def cpt_feature(env_state):
                    return self.feature.cpt_feature(env_state)

                return cpt_feature
        else:
            return feature_param

    def update_agent(self, agent):
        """update the agent in player

        Args:
            agent (BaseAgent): agent
        """
        self.agent_dict[agent.agent_id] = agent

    def build_agent(self, config):
        # TODO: move to every player
        """create agents from player config

        Args:
            config (dict): [description]

        Returns:
            dict: agents dict
        """
        agents = {}
        # for agent_id, agent_config in agent_configs.items():
        for agent_id in self.agent_list:
            agent_cls_str = self.config.player_config.agent_config[agent_id]
            if type(agent_cls_str) is str:
                if ":" in agent_cls_str:
                    agent_cls_str = "malib." + agent_cls_str
                    cls = load(agent_cls_str)
                else:
                    assert self.register_handle is not None
                    cls = ray.get(self.register_handle.get.remote())
            else:
                cls = agent_cls_str
            agent = cls(
                agent_id,
                config,
                training=self.training,
                register_handle=self.register_handle,
            )
            agents[agent_id] = agent
        return agents

    def reset_step_data(self):
        """clear the data saved"""
        # old code to be deleted
        self.step_data = {}
        # for i in self.agent_dict.keys():
        # self.step_data[i] = []

        # self.step_data_list=[]

    def reset(self):
        """reset the state of the player"""
        self.reset_step_data()

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
        raise notImplementedError

    def get_data(self):
        """get the one step data of the player

        Returns:
            Dict: one step data
        """
        return self.step_data

    def use_ref_weights(self):

        ref_w = list(self.ref_weights.values())
        while ref_w:
            w_get, ref_w = ray.wait(ref_w)
            train_result = ray.get(w_get[0])
            model_id = train_result["model_id"]
            self.set_weights(train_result["train_result"]["weights"], model_id)
        self.ref_weights = {}

    def set_weights(self, weights, model_id):
        """set the specified agent's model weights

        Args:
            weights (Dict): model weights
            agent_id (str): agent id
        """
        self.model_dict[model_id].set_weights(weights)

    def set_ref_weights(self, ref_weights, model_id, later=False):
        if later:
            self.ref_weights[model_id] = ref_weights[0]
        else:
            train_result = ray.get(ref_weights[0])
            weights = train_result["train_result"]["weights"]
            self.model_dict[model_id].set_weights(weights)

    def get_player_weights(self, model_id=None):
        """get the model weights of all agents in player

        Args:
            agent_id (str, optional): if agent_id is not None,get the specified agent's model weights. Defaults to None.

        Returns:
            dict: weights dict
        """
        # TODO:是否需要 定制 agent_id
        weights = {}
        for model_id, model in self.model_dict.items():
            weights[model_id] = model.get_weights()
        return weights

    def set_player_weights(self, player_weights):
        for m_id, m_w in player_weights.items():
            assert m_id in self.model_dict
            self.model_dict[m_id].set_weights(m_w)

    def update_player_name(self, p_name, p_long_name):
        """update the player info

        Args:
            p_name (str): player name
            p_long_name (str): player long name
        """
        self.player_name = p_name
        self.player_long_name = p_long_name
        self.info["player_name"] = p_name
        self.info["player_long_name"] = p_long_name

    def update_player_learn_step_num(self, learn_step_num):
        self.learn_step_number = learn_step_num
        self.info["learn_step_number"] = learn_step_num

    def make_model(self, config):
        model_dict = {}
        agent_config = config.player_config.agent_config
        model_id_set = set([agent_config[i].model_id for i in self.agent_list])
        for model_id in model_id_set:
            model_name = config.player_config.model_config[model_id].model_name
            model_params = config.player_config.model_config[model_id].model_params

            if type(model_name) is str:
                if ":" in model_name:
                    model_name = "malib." + model_name
                    cls = load(model_name)
                else:
                    assert self.register_handle is not None
                    cls = ray.get(self.register_handle.get.remote(model_name))
            else:
                cls = model_name
            model = cls(**model_params)

            model_dict[model_id] = model

        return model_dict

    def get_am_mapping(self):
        agent_config = self.config.player_config.agent_config
        self.agent2model = {}
        self.model2agent = collections.defaultdict(lambda: [])
        for agent_id in self.agent_list:
            model_id = agent_config[agent_id].model_id
            self.agent2model[agent_id] = model_id
            self.model2agent[model_id].append(agent_id)

    def make_mask(self, obs):
        return None
