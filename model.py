import torch
import torch.nn as nn
from pathlib import Path


class NNUE(nn.Module):

    # Piece-square features are replicated per own-king input bucket; the
    # bucket layout and mirroring live in fen_parser and must match
    # Weiawaga's nnue.rs. Indices arrive already bucketed.
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
            scale_grad_by_freq=True
        )
        self.input_layer_bias = nn.Parameter(torch.zeros(hyperparameters["L1"]))

        # Input factorizer: a bucket-independent embedding shared by all
        # king buckets, folded into each bucket's rows at save time. Lets
        # thin buckets generalize from the full dataset.
        self.virtual_input_layer = nn.EmbeddingBag(
            self.N_BASE_FEATURES + 1,
            hyperparameters["L1"],
            padding_idx=self.N_BASE_FEATURES,
            mode="sum",
            scale_grad_by_freq=True
        )
        nn.init.zeros_(self.virtual_input_layer.weight)

        self.hidden_layer = nn.ModuleList(
            [nn.Linear(2 * hyperparameters["L1"], 1) for _ in range(hyperparameters["N_BUCKETS"])]
        )

        self.virtual_hidden_layer = nn.Linear(2 * hyperparameters["L1"], 1)
        nn.init.zeros_(self.virtual_hidden_layer.weight)
        nn.init.zeros_(self.virtual_hidden_layer.bias)

        scale = self.N_BASE_FEATURES ** -0.5

        nn.init.uniform_(self.input_layer.weight, a=-scale, b=scale)
        nn.init.uniform_(self.input_layer_bias, a=-scale, b=scale)

    @classmethod
    def _virtual_indices(cls, indices: torch.Tensor) -> torch.Tensor:
        # Map bucketed indices onto the shared base features; the padding
        # index maps to the virtual layer's own padding row (a plain modulo
        # would collide it with base feature 0).
        return torch.where(
            indices == cls.N_FEATURES, cls.N_BASE_FEATURES, indices % cls.N_BASE_FEATURES
        )

    def _embed(self, indices: torch.Tensor) -> torch.Tensor:
        return (
            self.input_layer(indices)
            + self.virtual_input_layer(self._virtual_indices(indices))
            + self.input_layer_bias
        )

    def forward(self, batch: dict[str, torch.Tensor]):

        bucket_idx = (
            (batch["stm_indices"].ne(self.N_FEATURES).sum(dim=-1, keepdim=True) - 2)
            .div(-(32 // -len(self.hidden_layer)), rounding_mode="floor")
            .long()
        )

        stm_embeddings = self._embed(batch["stm_indices"].long())
        nstm_embeddings = self._embed(batch["nstm_indices"].long())

        embeddings = torch.cat([stm_embeddings, nstm_embeddings], dim=-1).clamp(0, 1).square()

        out = torch.cat([h(embeddings) for h in self.hidden_layer], dim=-1)
        out = out.gather(-1, bucket_idx) + self.virtual_hidden_layer(embeddings)

        return out

    def save_quantized(self, epoch: int, path: str) -> None:
        QA, QB = 255, 64

        def i16(t, factor):
            return torch.round(t * factor).to(torch.int16).flatten()

        # Fold the shared virtual rows into every king bucket's rows,
        # matching the engine's bucket-major input layout.
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

        with open(Path(path) / f"network_{epoch}.bin", "wb") as file:
            file.write(blob)


if __name__ == "__main__":
    from fen_parser import fen_to_indices

    (stm_indices, nstm_indices) = fen_to_indices("4k3/8/8/3q4/8/8/8/4K3 w - - 0 1")

    model = NNUE(L1=512, N_BUCKETS=8)

    input_batch = {
        "stm_indices": torch.tensor([stm_indices + [768] * (32 - len(stm_indices))], dtype=torch.long),
        "nstm_indices": torch.tensor([nstm_indices + [768] * (32 - len(nstm_indices))], dtype=torch.long),
    }

    # print(indices)

    print(model.psqt_layer(input_batch["stm_indices"].long()))
    print(model.psqt_layer(input_batch["nstm_indices"].long()))



    # print(400*model(input).item())

    # model_state_dict = torch.load("nets/current_network/model_checkpoint_49.pth")
    # print(model_state_dict["loss"])
    # print(f"{model_state_dict["loss"]:.2e}")

    # model = NNUE(L1=512, N_BUCKETS=8)
    #
    # model.load_state_dict(model_state_dict["model_state_dict"])
    #
    # model.save_quantized(1, "test_quantized")

    # stm_indices, nstm_indices = fen_to_indices("1r1r4/pp6/4kp2/4pQ1p/7q/8/6PP/5R1K b - - 1 33")
    # print(nstm_indices)
    #
    # input = {
    #     "stm_indices": torch.tensor([stm_indices + [768] * (32 - len(stm_indices))], dtype=torch.long),
    #     "nstm_indices": torch.tensor([nstm_indices + [768] * (32 - len(nstm_indices))], dtype=torch.long),
    # }
    #
    # print(400*model(input).item())
