import torch.nn as nn
import torch.nn.functional as F


class ScaledMSELoss(nn.Module):
    def __init__(self, lambda_: float, nnue_2_score: int, cp_scaling: int, **kwargs):
        super().__init__()
        self.lambda_ = lambda_
        self.nnue_2_score = nnue_2_score
        self.cp_scaling = cp_scaling

    def forward(self, pred, batch):
        wdl_pred = (pred * self.nnue_2_score / self.cp_scaling).sigmoid()
        wdl_cp = (batch["cp"] / self.cp_scaling).sigmoid()
        wdl_target = self.lambda_ * wdl_cp + (1 - self.lambda_) * batch["result"]

        return F.mse_loss(wdl_pred, wdl_target)


class ScaledCELoss(nn.Module):
    def __init__(self, lambda_: float, nnue_2_score: int, cp_scaling: int, **kwargs):
        super().__init__()
        self.lambda_ = lambda_
        self.nnue_2_score = nnue_2_score
        self.cp_scaling = cp_scaling

    def forward(self, pred, batch):
        wdl_cp = (batch["cp"] / self.cp_scaling).sigmoid()
        wdl_target = self.lambda_ * wdl_cp + (1 - self.lambda_) * batch["result"]

        return F.binary_cross_entropy_with_logits(
            pred * self.nnue_2_score / self.cp_scaling, wdl_target
        )
