import torch
import torch.nn as nn
from pathlib import Path


class NNUE(nn.Module):
    N_BASE_FEATURES = 768
    N_KING_BUCKETS = 4
    N_FEATURES = N_KING_BUCKETS * N_BASE_FEATURES

    def __init__(self, **hyperparameters):
        super().__init__()
        self.input_layer = nn.EmbeddingBag(
            self.N_FEATURES + 1,
            hyperparameters["L1"],
            padding_idx=self.N_FEATURES,
            mode="sum",
            scale_grad_by_freq=True,
        )
        self.input_layer_bias = nn.Parameter(torch.zeros(hyperparameters["L1"]))

        self.virtual_input_layer = nn.EmbeddingBag(
            self.N_BASE_FEATURES + 1,
            hyperparameters["L1"],
            padding_idx=self.N_BASE_FEATURES,
            mode="sum",
            scale_grad_by_freq=True,
        )
        nn.init.zeros_(self.virtual_input_layer.weight)

        self.hidden_layer = nn.ModuleList(
            [
                nn.Linear(2 * hyperparameters["L1"], 1)
                for _ in range(hyperparameters["N_BUCKETS"])
            ]
        )

        self.virtual_hidden_layer = nn.Linear(2 * hyperparameters["L1"], 1)
        nn.init.zeros_(self.virtual_hidden_layer.weight)
        nn.init.zeros_(self.virtual_hidden_layer.bias)

        # Post-hoc WDL readout on the frozen trunk. Registered last so the
        # optimizer state of older eval-only checkpoints still aligns.
        self.wdl_layer = nn.Linear(2 * hyperparameters["L1"], 3)

        scale = self.N_BASE_FEATURES**-0.5

        nn.init.uniform_(self.input_layer.weight, a=-scale, b=scale)
        nn.init.uniform_(self.input_layer_bias, a=-scale, b=scale)

    @classmethod
    def _virtual_indices(cls, indices: torch.Tensor) -> torch.Tensor:
        # Map bucketed indices onto the shared base features; the padding
        # index maps to the virtual layer's own padding row (a plain modulo
        # would collide it with base feature 0).
        return torch.where(
            indices == cls.N_FEATURES,
            cls.N_BASE_FEATURES,
            indices % cls.N_BASE_FEATURES,
        )

    def _embed(self, indices: torch.Tensor) -> torch.Tensor:
        return (
            self.input_layer(indices)
            + self.virtual_input_layer(self._virtual_indices(indices))
            + self.input_layer_bias
        )

    def _features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        stm_embeddings = self._embed(batch["stm_indices"].long())
        nstm_embeddings = self._embed(batch["nstm_indices"].long())

        return torch.cat([stm_embeddings, nstm_embeddings], dim=-1).clamp(0, 1).square()

    def forward(self, batch: dict[str, torch.Tensor]):

        bucket_idx = (
            (batch["stm_indices"].ne(self.N_FEATURES).sum(dim=-1, keepdim=True) - 2)
            .div(-(32 // -len(self.hidden_layer)), rounding_mode="floor")
            .long()
        )

        embeddings = self._features(batch)

        out = torch.cat([h(embeddings) for h in self.hidden_layer], dim=-1)
        out = out.gather(-1, bucket_idx) + self.virtual_hidden_layer(embeddings)

        return out

    def forward_wdl(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Trunk under no_grad: gradients only ever reach the WDL head.
        with torch.no_grad():
            features = self._features(batch)
        return self.wdl_layer(features)

    def save_quantized(self, epoch: int, path: str) -> None:
        QA, QB = 255, 64

        def i16(t, factor):
            return torch.round(t * factor).to(torch.int16).flatten()

        input_weight = self.input_layer.weight[: self.N_FEATURES]
        virtual_weight = self.virtual_input_layer.weight[: self.N_BASE_FEATURES]
        folded = input_weight + virtual_weight.repeat(self.N_KING_BUCKETS, 1)

        weights = [
            i16(folded, QA),
            i16(self.input_layer_bias, QA),
        ]
        biases = []
        for layer in self.hidden_layer:
            weights.append(i16(layer.weight + self.virtual_hidden_layer.weight, QB))
            biases.append(i16(layer.bias + self.virtual_hidden_layer.bias, QB))

        blob = torch.cat(weights).cpu().numpy().astype("<i2").tobytes()
        blob += torch.cat(biases).cpu().numpy().astype("<i2").tobytes()

        # WDL head, stored unquantized: the engine only evaluates it when
        # printing info lines, so it reads plain f32 logits.
        blob += self.wdl_layer.weight.detach().cpu().numpy().astype("<f4").tobytes()
        blob += self.wdl_layer.bias.detach().cpu().numpy().astype("<f4").tobytes()

        with open(Path(path) / f"network_{epoch}.bin", "wb") as file:
            file.write(blob)
