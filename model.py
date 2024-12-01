import os
import torch
import torch.nn as nn


class Clamp(nn.Module):

    @staticmethod
    def forward(x):
        return x.clamp(0, 1)


class NNUE(nn.Module):

    def __init__(self, **hyperparameters):
        super().__init__()
        self.input_layer = nn.Linear(768, hyperparameters["L1"])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hyperparameters["L1"], 1) for _ in range(8)]
        )
        self.psqt_layer = nn.Linear(768, 1, bias=False)

    def forward(self, x):
        bucket_idx = (
            (x.sum(dim=-1, keepdim=True) - 1).div(4, rounding_mode="floor").long()
        )

        out = self.input_layer(x)
        out = out.clamp(0, 1)
        out = torch.cat([h(out) for h in self.hidden_layers], dim=-1).gather(
            -1, bucket_idx
        )

        return out + self.psqt_layer(x)

    def save_quantized(self, epoch, path):
        with open(os.path.join(path, f"model_parameters_{epoch}.rs"), "w") as file:
            for name, tensor in self.state_dict().items():
                name = name.replace("net.", "").replace(".", "_").upper()

                if tensor.dim() >= 2:
                    tensor = tensor.transpose(0, 1)

                factor = 255 if "INPUT" in name else 64
                tensor = self.quantize(tensor, factor).flatten()
                file.write(
                    f"pub const {name}: [i16; {len(tensor)}] = {tensor.tolist()};\n\n"
                )

    @staticmethod
    def quantize(x, factor):
        return (x * factor).round().to(torch.int16)
