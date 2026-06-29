import torch
import torch.nn as nn
from pathlib import Path


class NNUE(nn.Module):

    N_FEATURES = 768

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

        self.hidden_layer = nn.ModuleList(
            [nn.Linear(2 * hyperparameters["L1"], 1) for _ in range(hyperparameters["N_BUCKETS"])]
        )

        scale = self.N_FEATURES ** -0.5

        nn.init.uniform_(self.input_layer.weight, a=-scale, b=scale)
        nn.init.uniform_(self.input_layer_bias, a=-scale, b=scale)

    def forward(self, batch: dict[str, torch.Tensor]):

        bucket_idx = (
            (batch["stm_indices"].ne(self.N_FEATURES).sum(dim=-1, keepdim=True) - 2)
            .div(-(32 // -len(self.hidden_layer)), rounding_mode="floor")
            .long()
        )

        stm_embeddings = self.input_layer(batch["stm_indices"].long()) + self.input_layer_bias
        nstm_embeddings = self.input_layer(batch["nstm_indices"].long()) + self.input_layer_bias

        embeddings = torch.cat([stm_embeddings, nstm_embeddings], dim=-1).clamp(0, 1).square()

        out = torch.cat([h(embeddings) for h in self.hidden_layer], dim=-1).gather(
            -1, bucket_idx
        )

        return out

    def save_quantized(self, epoch: int, path: str) -> None:
        # Emit a raw little-endian blob the engine embeds via include_bytes!:
        # all i16 weight sections first (each stays 16-aligned so the Rust side
        # can reinterpret them as i16x16), then the i16 bias section. Input
        # weights/bias are scale QA; hidden weights/biases are scale QB. Section
        # order must match Network::new in nnue.rs.
        QA, QB = 255, 64

        def i16(t, factor):
            return torch.round(t * factor).to(torch.int16).flatten()

        weights = [
            i16(self.input_layer.weight[: self.N_FEATURES], QA),  # drop padding row
            i16(self.input_layer_bias, QA),
        ]
        biases = []
        for layer in self.hidden_layer:
            weights.append(i16(layer.weight, QB))
            biases.append(i16(layer.bias, QB))

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
