import torch
import torch.nn as nn


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()

    def get_weights(self):
        return self.state_dict()

    def set_weights(self, weights):
        self.load_state_dict(weights)


class DataParallel(nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__(
            module, device_ids=device_ids, output_device=output_device, dim=dim
        )

    def get_weights(self):
        result = dict(self.state_dict())
        result = {k.replace("module.", ""): v for k, v in result.items()}
        return result

    def set_weights(self, weights):
        weights = {"module." + k: v for k, v in weights.items()}
        self.load_state_dict(weights)
