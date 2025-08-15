import os
import torch
import torch.nn as nn


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

        def quantize(x, factor):
            return x.mul(factor).round().to(torch.int16)

        def chunk_into_i16xn(lst: list[int], n: int) -> list[str]:
            """
            Convert a list of integers into SIMD-style i16xN chunks for Rust.

            Parameters:
                lst (list[int]): Flat list of quantized i16 values.
                n (int): The SIMD width (e.g., 8, 16).

            Returns:
                list[str]: A list of strings like `i16xN::new([...])`
            """
            if len(lst) % n != 0:
                lst += [0] * (n - len(lst) % n)

            return [
                f"i16x{n}::new([{', '.join(str(value) for value in lst[i: i + n])}])"
                for i in range(0, len(lst), n)
            ]

        file_path = os.path.join(path, f"model_parameters_{epoch}.rs")
        with open(file_path, "w") as file:
            file.write("use wide::i16x16;\n\n")

            # ------------------------------------------------------------------
            # Input-layer embedding weights (2-D)
            # ------------------------------------------------------------------
            embed_name = "INPUT_LAYER_WEIGHT"
            embed_tensor = self.input_layer.weight[: self.N_FEATURES]         # drop padding row
            embed_factor = 255
            embed_quant = quantize(embed_tensor, embed_factor).tolist()

            n_rows = len(embed_quant)
            n_cols = len(embed_quant[0]) if n_rows else 0
            file.write(
                f"pub static {embed_name}: [[i16x16; {(n_cols + 15) // 16}]; {n_rows}] = [\n"
            )
            for row in embed_quant:
                chunks = chunk_into_i16xn(list(row), 16)
                file.write(f"    [{', '.join(chunks)}],\n")
            file.write("];\n\n")

            # ------------------------------------------------------------------
            # Input-layer bias (1-D)
            # ------------------------------------------------------------------
            bias_name = "INPUT_LAYER_BIAS"
            bias_factor = 255
            bias_quant = quantize(self.input_layer_bias, bias_factor).tolist()
            bias_chunks = chunk_into_i16xn(bias_quant, 16)
            file.write(
                f"pub static {bias_name}: [i16x16; {len(bias_chunks)}] = "
                f"[{', '.join(bias_chunks)}];\n\n"
            )

            # ------------------------------------------------------------------
            # Hidden buckets (weights & biases)
            # ------------------------------------------------------------------
            for idx, layer in enumerate(self.hidden_layer):
                # Weights
                w_name = f"HIDDEN_LAYER_{idx}_WEIGHT"
                w_factor = 64
                w_quant = quantize(layer.weight, w_factor).flatten().tolist()
                w_chunks = chunk_into_i16xn(w_quant, 16)
                file.write(
                    f"pub static {w_name}: [i16x16; {len(w_chunks)}] = "
                    f"[{', '.join(w_chunks)}];\n\n"
                )

                # Bias (single scalar per bucket)
                b_name = f"HIDDEN_LAYER_{idx}_BIAS"
                b_factor = 64
                b_values = quantize(layer.bias, b_factor).flatten().tolist()
                file.write(
                    f"pub static {b_name}: [i16; {len(b_values)}] = "
                    f"[{', '.join(str(value) for value in b_values)}];\n\n"
                )


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
