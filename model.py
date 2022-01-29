import torch
import torch.nn as nn

INPUT_FEATURES = 768
L1 = 128
CP_SCALING = 0.00837018


class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(INPUT_FEATURES, L1)
        self.hidden_layer = nn.Linear(L1, 1)

    def forward(self, inputs):
        out = torch.clamp(self.input_layer(inputs), min=0, max=1)
        out = self.hidden_layer(out)
        return out


def loss(lambda_, pred, score, game_result):
    wdl_eval_model = (pred * CP_SCALING).sigmoid()
    wdl_eval_target = (score * CP_SCALING).sigmoid()

    wdl_value_target = lambda_ * wdl_eval_target + (1 - lambda_) * game_result

    return (wdl_eval_model - wdl_value_target).square().mean()


if __name__ == "__main__":
    import json
    from fen_parser import fen_to_vec

    # torch.set_printoptions(profile="full")
    model = NNUE()
    model.load_state_dict(torch.load("nets/lambda1.0_epochs20_lr0.0005_bs8192_2022-01-27T08-47-04/model_state_dict_20.pth"))

    model.eval()
    print(model)
    print(model(torch.Tensor(fen_to_vec("q7/4BQ2/7k/4p3/8/6P1/5P1K/8 w - - 2 61"))))
