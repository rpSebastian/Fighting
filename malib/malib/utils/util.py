import sys
import importlib
import importlib.abc, importlib.util
import ray


def load(name):

    if ':' in name:
        mod_name, attr_name = name.split(":")
    else:
        li=name.split(".")
        mod_name, attr_name = '.'.join(li[:-1]),li[-1]

    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def make_model(config, agent_id, register_handle=None):
    model_name = config.agent_config[agent_id].model_name
    input_shape = config.agent_config[agent_id].model_in_shape
    output_shape = config.agent_config[agent_id].model_out_shape
    if type(model_name) is str:
        if ":" in model_name:
            model_name = "malib." + model_name
            cls = load(model_name)
        else:
            assert register_handle is not None
            cls = ray.get(register_handle.get.remote(model_name))
    else:
        cls = model_name
    model = cls(in_dim=input_shape, out_dim=output_shape)
    return model


class StringLoader(importlib.abc.SourceLoader):
    def __init__(self, data):
        self.data = data

    def get_source(self, fullname):
        return self.data

    def get_data(self, path):
        return self.data.encode("utf-8")

    def get_filename(self, fullname):
        return "<not a real path>/" + fullname + ".py"


def loadString(string_name):
    module_name = string_name.strip(".py")
    with open(string_name, "r") as module:
        loader = StringLoader(module.read())

    spec = importlib.util.spec_from_loader(module_name, loader, origin="built-in")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
