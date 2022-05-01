import config
import torch
import torch.nn as nn


class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(config.INPUT_FEATURES, config.L1)
        self.hidden_layer = nn.Linear(config.L1, config.N_BUCKETS)

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
        wdl_eval_model = (pred * config.CP_SCALING).sigmoid()
        wdl_eval_target = (score * config.CP_SCALING).sigmoid()
        wdl_value_target = (
            config.LAMBDA * wdl_eval_target + (1 - config.LAMBDA) * game_result
        )

        loss = (
            wdl_value_target * torch.log(wdl_value_target + config.EPSILON)
            + (1 - wdl_value_target) * torch.log(1 - wdl_value_target + config.EPSILON)
        ) - (
            wdl_value_target * torch.log(wdl_eval_model + config.EPSILON)
            + (1 - wdl_value_target) * torch.log(1 - wdl_eval_model + config.EPSILON)
        )

        return loss.mean()
