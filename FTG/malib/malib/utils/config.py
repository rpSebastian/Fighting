import os
import copy
import datetime
from malib.utils import Logger
from malib.utils import loadString


class BaseConfig(object):
    def __init__(self, config_dict=None):
        if config_dict:
            self._update(config_dict)

    def _update(self, data_dict: dict):
        for k, v in data_dict.items():
            if type(v) is dict:
                if k in self.__dict__.keys() and type(self.__dict__[k]) is BaseConfig:
                    self.__dict__[k]._update(v)
                else:
                    if k not in ["env_params", "model_params"]:
                        self.__dict__[k] = BaseConfig(v)
                    else:
                        self.__dict__[k] = v
            else:
                self.__dict__[k] = v

    def __getitem__(self, key):
        return self.__dict__[key]

    def _convert_to_dict(self, base_config):
        for k, v in base_config.items():
            if type(v) is BaseConfig:
                base_config[k] = self._convert_to_dict(v.__dict__)
        return base_config

    def get_config_dict(self):
        config_dict = copy.deepcopy(self.__dict__)
        config_dict = self._convert_to_dict(config_dict)

        return config_dict

    def has_key(self, key):
        if key in self.__dict__.keys():
            return True
        else:
            return False

    def __contains__(self, key):
        return self.has_key(key)


class Config(BaseConfig):
    def __init__(self, config_dict=None):
        if config_dict is not None:
            for k, v in config_dict.items():
                self.__dict__[k] = v

    def update(self, config_dict):
        if (
            config_dict["config_name"].startswith("learne")
            or config_dict["config_name"] == "league_config"
        ):
            self._update(config_dict)
        else:
            config_name = config_dict["config_name"]
            if config_name not in self.__dict__.keys():
                self.__dict__[config_dict["config_name"]] = BaseConfig()
            self.__dict__[config_dict["config_name"]]._update(config_dict)

    def dict2linelist(self, d, level, config_name=None, ident="    "):
        result = []
        if config_name:
            line = config_name + "=dict("
            result.append(line)
        assert type(d) is dict
        for k, v in d.items():
            if type(v) is dict:
                line = ident * level + k + "=dict("
                result.append(line)
                result.extend(self.dict2linelist(v, level + 1))
                result.append(ident * level + "),")
            else:
                v = v if type(v) is not str else '"' + v + '"'
                line = ident * level + k + "=" + str(v) + ","
                result.append(line)
        if config_name:
            line = ")"
            result.append(line)
        return result

    def dict2lines(self):
        d = self.get_config_dict()
        config_name = d["config_name"]
        result = self.dict2linelist(d, level=1, config_name=config_name)
        return result

    def save(self, name=None):
        path = os.path.join(self.log_config.log_dir, self.log_config.time_now)
        if not os.path.exists(path):
            os.makedirs(path)
        if name:
            path = path + "/" + name + "_config.py"
        else:
            path = path + "/config.py"
        result = self.dict2lines()
        with open(path, "a") as f:
            for r in result:
                f.write(r)
                f.write("\n")

    def load_config(self, file):
        config_obj = loadString(file)
        if self.config_name in config_obj.__dict__.keys():
            config = config_obj.__dict__[self.config_name]
            self.update(config)
