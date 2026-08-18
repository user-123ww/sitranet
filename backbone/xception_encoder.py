import torch
import torch.nn as nn

from .xception import Xception


class XceptionEncoder(nn.Module):
    def __init__(self, pretrained_path='backbone/xception-b5690688.pth'):
        super().__init__()
        config = {
            'num_classes': 2,
            'mode': 'adjust_channel',
            'inc': 3,
            'dropout': False,
        }
        self.xception = Xception(config)

        pretrained_dict = torch.load(pretrained_path, weights_only=True)
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']

        model_dict = self.xception.state_dict()
        compatible_weights = {}
        for name, weights in pretrained_dict.items():
            if 'fc.weight' in name or 'fc.bias' in name:
                continue
            if 'pointwise' in name and weights.ndim == 2:
                compatible_weights[name] = weights.unsqueeze(-1).unsqueeze(-1)
            elif name in model_dict and model_dict[name].size() == weights.size():
                compatible_weights[name] = weights

        self.xception.load_state_dict(compatible_weights, strict=False)

    def forward(self, x):
        return self.xception.features(x)
