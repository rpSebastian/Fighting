import sys
import importlib.abc, importlib.util


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