import os
import torch
import torch.nn as nn

from collections import OrderedDict


class Clamp(nn.Module):

    @staticmethod
    def forward(x):
        return x.clamp(0, 1)


class NNUE(nn.Module):

    def __init__(self, L1):
        super().__init__()

        self.net = nn.Sequential(
            OrderedDict(
                [
                    ("input_layer", nn.Linear(768, L1)),
                    ("clamp1", Clamp()),
                    ("l1", nn.Linear(L1, 1)),
                ]
            )
        )

    def forward(self, x):
        return self.net(x)

    def checkpoint(self, epoch, path, scale):

        torch.save(
            self.state_dict(), os.path.join(path, f"model_state_dict_{epoch}.pth")
        )

        param_dict = {
            name: torch.flatten(
                tensor.T if "input" in name or "residual" in name else tensor
            )
            for name, tensor in self.state_dict().items()
        }

        with open(os.path.join(path, f"model_parameters_{epoch}.rs"), "w") as file:
            for name, param in param_dict.items():
                name = name.replace("net.", "")
                if name == "l1.bias":
                    param = param.mul(scale)
                param = param.mul(scale).round().to(torch.int16)
                file.write(
                    f"pub const {name.replace('.', '_').upper()}: [i16; {len(param)}] = {param.tolist()};\n\n"
                )
