import ray


@ray.remote
class Registry:
    def __init__(self):
        print("Registry init ...")
        self.storage = {}

    def add(self, str_id, obj):
        assert str_id not in self.storage
        self.storage[str_id] = obj

    def get(self, str_id):
        return self.storage[str_id]

    def get_names(self):
        return self.__dict__.keys()


registry = None


def regist(id: str, obj):
    assert registry is not None
    registry.add.remote(id, obj)


def register_init():
    global registry
    registry = Registry.remote()
    return registry


def get(register_handle, id: str):
    return ray.get(register_handle.get.remote(id))