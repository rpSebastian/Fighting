# import importlib

from malib.utils import load

from .env import BaseEnv


class EnvRegistry(object):
    def __init__(self):
        self.envs = {}

    def register(self, id: str, params: dict) -> None:
        if id in self.envs:
            # return
            # raise Exception("{} has already registed".format(id))
            # TODO: find better method to register Env
            pass
        self.envs[id] = params

    def make(self, id: str, params=None) -> BaseEnv:
        if not params:
            params = self.envs[id]
        raw_env = params["raw_env"]
        private_env = False
        env_dir = None
        if "env_dir" in params:
            private_env = True
            env_dir = params["env_dir"] + "."
            if "wrapper_dir" in params:
                wrapper_dir = params["wrapper_dir"]
            else:
                wrapper_dir = env_dir
        else:
            env_dir = "malib.aienv.env."
            wrapper_dir = "malib.aienv.wrapper."

        if type(raw_env) is str:
            if ":" in raw_env or private_env:
                raw_env = env_dir + raw_env
                e_cls = load(raw_env)
            else:
                raise ValueError("not implemented")
        else:
            e_cls = raw_env
        if "env_params" in params:
            env_params = params["env_params"]
            if type(env_params) is not dict:
                env_params = env_params.get_config_dict()
            env = e_cls(env_params)
        else:
            env = e_cls()
        if "wrapper" in params:
            for i, w in enumerate(params["wrapper"]):
                if type(w) is str:
                    if ":" in w or private_env:
                        if type(wrapper_dir) is str:
                            raw_wrapper_name = wrapper_dir + w
                        else:
                            raw_wrapper_name = wrapper_dir[i] + w
                        w_cls = load(raw_wrapper_name)
                    else:
                        raise ValueError("wrapper error")
                else:
                    w_cls = w
                env = w_cls(env)

        return env


registry = EnvRegistry()

count = 0


def register(id: str, params: dict) -> None:
    registry.register(id, params)


def make(id: str, params=None) -> BaseEnv:
    return registry.make(id, params)
