import torch
import torch.nn as nn

INPUT_FEATURES = 768
L1 = 256
CP_SCALING = 0.008380556589999985
N_BUCKETS = 8


class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(INPUT_FEATURES, L1)
        self.hidden_layer = nn.Linear(L1, N_BUCKETS)

        self.idx_offset = None

    def forward(self, inputs):

        buckets = (inputs.sum(dim=1) - 1).div(4, rounding_mode="floor").long()
        if self.idx_offset is None or self.idx_offset.shape[0] != inputs.shape[0]:
            self.idx_offset = torch.arange(0, inputs.shape[0] * N_BUCKETS, N_BUCKETS, device=inputs.device)

        indices = buckets.flatten() + self.idx_offset
        out = self.input_layer(inputs)
        out = torch.clamp(out, min=0, max=1)

        out = self.hidden_layer(out)
        out = out.view(-1, 1)[indices]
        return out


def loss(lambda_, pred, score, game_result):
    wdl_eval_model = (pred * CP_SCALING).sigmoid()
    wdl_eval_target = (score * CP_SCALING).sigmoid()

    wdl_value_target = lambda_ * wdl_eval_target + (1 - lambda_) * game_result

    return (wdl_eval_model - wdl_value_target).square().mean()


if __name__ == "__main__":
    import json
    from fen_parser import fen_to_vec

    model = NNUE()
    model.load_state_dict(
        torch.load(
            "nets/bucket_psqt_lambda0.8_lr0.001_bs8192_2022-02-04T04-29-17/model_state_dict_19.pth",
            map_location="cpu"
        )
    )

    model.eval()
    print(model)
    vec = fen_to_vec("3r4/p3r3/1k6/2p5/2p4p/3B4/PPP2QPP/3K4 w - - 0 38")
    with torch.no_grad():
        print(model(torch.Tensor([vec])))

    # torch.set_printoptions(threshold=10_000_000)
    # print(torch.round(model.state_dict()["input_layer.weight"].flatten()*64).round().to(int))
