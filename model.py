import os
import torch
import torch.nn as nn

import math

class CustomFakeQuant(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quantized = torch.round(x * self.factor) / self.factor
        # Use STE: Pass fake quant output but use original gradient for backward pass
        return x + (quantized - x).detach()


class NNUE(nn.Module):

    N_FEATURES = 768

    def __init__(self, **hyperparameters):
        super().__init__()
        self.input_layer = nn.EmbeddingBag(self.N_FEATURES + 1, hyperparameters["L1"], padding_idx=self.N_FEATURES, mode="sum", scale_grad_by_freq=True)
        self.input_layer_bias = nn.Parameter(torch.zeros(hyperparameters["L1"]))

        self.hidden_layer = nn.ModuleList(
            [nn.Linear(2 * hyperparameters["L1"], 1) for _ in range(8)]
        )

        scale = 1 / math.sqrt(self.N_FEATURES)

        nn.init.uniform_(self.input_layer.weight, a=-scale, b=scale)
        nn.init.uniform_(self.input_layer_bias, a=-scale, b=scale)

    def forward(self, batch):

        bucket_idx = (
            (batch["stm_indices"].ne(self.N_FEATURES).sum(dim=-1, keepdim=True) - 2)
            .div(4, rounding_mode="floor")
            .long()
        )

        stm_embeddings = self.input_layer(batch["stm_indices"]) + self.input_layer_bias
        nstm_embeddings = self.input_layer(batch["nstm_indices"]) + self.input_layer_bias

        embeddings = torch.cat([stm_embeddings, nstm_embeddings], dim=-1).clamp(0, 1)

        out = torch.cat([h(embeddings) for h in self.hidden_layer], dim=-1).gather(
            -1, bucket_idx
        )

        return out

    def save_quantized(self, epoch, path):
        with open(os.path.join(path, f"model_parameters_{epoch}.rs"), "w") as file:
            for name, tensor in self.state_dict().items():
                name = name.replace("net.", "").replace(".", "_").upper()

                # Determine quantization factor
                factor = 255 if "INPUT" in name else 64
                tensor = self.quantize(tensor[:self.N_FEATURES], factor)

                if "INPUT_LAYER_WEIGHT" in name:
                    # Quantize up to N_FEATURES only
                    nested = tensor.tolist()
                    outer_len = len(nested)
                    inner_len = len(nested[0]) if nested else 0

                    file.write(
                        f"pub const {name}: [[i16; {inner_len}]; {outer_len}] = [\n"
                    )
                    for row in nested:
                        file.write(f"    {row},\n")
                    file.write("];\n\n")

                else:
                    # Handle bias and other tensors as flat
                    tensor = tensor.flatten()
                    file.write(
                        f"pub const {name}: [i16; {len(tensor)}] = {tensor.tolist()};\n\n"
                    )

    @staticmethod
    def quantize(x, factor):
        return x.mul(factor).round().to(torch.int16)

