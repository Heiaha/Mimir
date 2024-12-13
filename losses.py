import torch.nn as nn


class ScaledMSELoss(nn.Module):
    def __init__(self, lambda_: float, nnue_2_score: int, cp_scaling: int, **kwargs):
        super().__init__()
        self.lambda_ = lambda_
        self.nnue_2_score = nnue_2_score
        self.cp_scaling = cp_scaling

    def forward(self, pred, cp, result):
        wdl_pred = (pred * self.nnue_2_score / self.cp_scaling).sigmoid()
        wdl_cp = (cp / self.cp_scaling).sigmoid()

        wdl_target = self.lambda_ * wdl_cp + (1 - self.lambda_) * result

        return (wdl_pred - wdl_target).square().mean()


class ScaledCELoss(nn.Module):

    EPSILON = 1e-12

    def __init__(self, lambda_: float, nnue_2_score: int, cp_scaling: int, **kwargs):
        super().__init__()
        self.lambda_ = lambda_
        self.nnue_2_score = nnue_2_score
        self.cp_scaling = cp_scaling

    def forward(self, pred, cp, result):
        wdl_pred = (pred * self.nnue_2_score / self.cp_scaling).sigmoid()
        wdl_cp = (cp / self.cp_scaling).sigmoid()

        wdl_target = self.lambda_ * wdl_cp + (1 - self.lambda_) * result

        return (
            (
                wdl_target * (wdl_target + self.EPSILON).log()
                + (1 - wdl_target) * (1 - wdl_target + self.EPSILON).log()
            )
            - (
                wdl_target * (wdl_pred + self.EPSILON).log()
                + (1 - wdl_target) * (1 - wdl_pred + self.EPSILON).log()
            )
        ).mean()
