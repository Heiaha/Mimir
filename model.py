import torch
import torch.nn as nn
from pathlib import Path


class NNUE(nn.Module):
    N_BASE_FEATURES = 768
    N_KING_BUCKETS = 4
    N_FEATURES = N_KING_BUCKETS * N_BASE_FEATURES

    def __init__(self, counts_per_batch=None, **hyperparameters):
        super().__init__()
        self.input_layer = nn.EmbeddingBag(
            self.N_FEATURES,
            hyperparameters["L1"],
            mode="sum",
            scale_grad_by_freq=False,
        )
        self.input_layer_bias = nn.Parameter(torch.zeros(hyperparameters["L1"]))

        self.virtual_input_layer = nn.EmbeddingBag(
            self.N_BASE_FEATURES,
            hyperparameters["L1"],
            mode="sum",
            scale_grad_by_freq=False,
        )
        nn.init.zeros_(self.virtual_input_layer.weight)

        self.hidden_layer = nn.Linear(
            2 * hyperparameters["L1"], hyperparameters["N_BUCKETS"]
        )

        self.virtual_hidden_layer = nn.Linear(2 * hyperparameters["L1"], 1)
        nn.init.zeros_(self.virtual_hidden_layer.weight)
        nn.init.zeros_(self.virtual_hidden_layer.bias)

        self.wdl_layer = nn.Linear(2 * hyperparameters["L1"], 3)

        scale = self.N_BASE_FEATURES**-0.5

        nn.init.uniform_(self.input_layer.weight, a=-scale, b=scale)
        nn.init.uniform_(self.input_layer_bias, a=-scale, b=scale)

        if counts_per_batch is not None:
            def damping(expected):
                return expected.clamp_min(1.0).reciprocal().unsqueeze(1)

            self.register_buffer(
                "input_damping", damping(counts_per_batch), persistent=False
            )
            self.register_buffer(
                "virtual_damping",
                damping(
                    counts_per_batch.view(self.N_KING_BUCKETS, -1).sum(0)
                ),
                persistent=False,
            )
            self.input_layer.weight.register_hook(lambda g: g * self.input_damping)
            self.virtual_input_layer.weight.register_hook(
                lambda g: g * self.virtual_damping
            )

    def _embed(self, flat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return (
            self.input_layer(flat, offsets)
            + self.virtual_input_layer(flat % self.N_BASE_FEATURES, offsets)
            + self.input_layer_bias
        )

    def _features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        stm_embeddings = self._embed(batch["stm_flat"], batch["offsets"])
        nstm_embeddings = self._embed(batch["nstm_flat"], batch["offsets"])

        return torch.cat([stm_embeddings, nstm_embeddings], dim=-1).clamp(0, 1).square()

    def forward(self, batch: dict[str, torch.Tensor]):
        bucket_idx = (
            (batch["counts"].unsqueeze(-1) - 2)
            .div(-(32 // -self.hidden_layer.out_features), rounding_mode="floor")
            .long()
        )

        embeddings = self._features(batch)

        out = self.hidden_layer(embeddings).gather(-1, bucket_idx)
        out = out + self.virtual_hidden_layer(embeddings)

        return out

    def forward_wdl(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            features = self._features(batch)
        return self.wdl_layer(features)

    def save_quantized(self, epoch: int, path: str) -> None:
        QA, QB = 255, 64

        def i16(t, factor):
            return torch.round(t * factor).to(torch.int16).flatten()

        folded = self.input_layer.weight + self.virtual_input_layer.weight.repeat(
            self.N_KING_BUCKETS, 1
        )

        # Row-major flattening gives the engine its bucket-major weight order.
        weights = [
            i16(folded, QA),
            i16(self.input_layer_bias, QA),
            i16(self.hidden_layer.weight + self.virtual_hidden_layer.weight, QB),
        ]
        bias = i16(self.hidden_layer.bias + self.virtual_hidden_layer.bias, QB)

        blob = torch.cat(weights).cpu().numpy().astype("<i2").tobytes()
        blob += bias.cpu().numpy().astype("<i2").tobytes()

        # The WDL head is stored unquantized; the engine reads it as f32.
        blob += self.wdl_layer.weight.detach().cpu().numpy().astype("<f4").tobytes()
        blob += self.wdl_layer.bias.detach().cpu().numpy().astype("<f4").tobytes()

        with open(Path(path) / f"network_{epoch}.bin", "wb") as file:
            file.write(blob)
