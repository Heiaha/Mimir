import torch
import torch.nn as nn


INPUT_FEATURES = 768
L1 = 256
CP_SCALING = 0.007828325269999983
N_BUCKETS = 8
LAMBDA = 1.0

EPSILON = 1e-12


class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(INPUT_FEATURES, L1)
        self.hidden_layer = nn.Linear(L1, N_BUCKETS)

    def forward(self, inputs):

        buckets = (
            (inputs.sum(dim=1) - 1)
            .div(4, rounding_mode="floor")
            .unsqueeze(dim=1)
            .long()
        )

        out = self.input_layer(inputs)
        out = torch.clamp(out, min=0, max=1)

        out = self.hidden_layer(out)
        out = out.gather(1, buckets)

        return out

    @staticmethod
    def loss(pred, score, game_result):
        wdl_eval_model = (pred * CP_SCALING).sigmoid()
        wdl_eval_target = (score * CP_SCALING).sigmoid()
        wdl_value_target = (
            LAMBDA * wdl_eval_target + (1 - LAMBDA) * game_result
        )

        loss = (
            wdl_value_target * torch.log(wdl_value_target + EPSILON)
            + (1 - wdl_value_target) * torch.log(1 - wdl_value_target + EPSILON)
        ) - (
            wdl_value_target * torch.log(wdl_eval_model + EPSILON)
            + (1 - wdl_value_target) * torch.log(1 - wdl_eval_model + EPSILON)
        )

        return loss.mean()


if __name__ == "__main__":
    import json
    from fen_parser import fen_to_vec

    model = NNUE()
    model.load_state_dict(
        torch.load(
            "nets/bucket_lambda0.5_lr0.001_bs8192_2022-03-08T02-01-41/model_state_dict_7.pth",
            map_location="cpu",
        )
    )

    model.eval()
    print(model)
    vec = fen_to_vec("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    with torch.no_grad():
        print(model(torch.Tensor([vec])))

    # torch.set_printoptions(threshold=10_000_000)
    # print(torch.round(model.state_dict()["psqt_layer.weight"].T.flatten()*64).round().to(int))
