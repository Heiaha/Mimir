import config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict

from dataset import Batch


class NNUE(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(config.L0, config.L1)
        self.l1 = nn.Linear(config.L1, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=0, verbose=True, min_lr=1e-6
        )
        self.to(config.DEVICE)

    def forward(self, X):

        l0_ = self.input_layer(X).clamp(0, 1)
        return self.l1(l0_)

    @staticmethod
    def loss(pred, cp, result):
        wdl_pred = (pred * config.NNUE_2_SCORE / config.CP_SCALING).sigmoid()
        wdl_cp = (cp / config.CP_SCALING).sigmoid()

        wdl_target = config.LAMBDA * wdl_cp + (1 - config.LAMBDA) * result

        return (wdl_pred - wdl_target).square().mean()

    def step(self, batch):
        pred = self(batch.X)
        loss = self.loss(pred, batch.cp, batch.result)
        loss.backward()
        self.optimizer.step()
        # zero the parameter gradients
        self.optimizer.zero_grad()
        return loss

    def val_loss(self, val_data):
        self.eval()
        with torch.no_grad():
            running_loss = 0
            n_batches = 0
            for batch_data in val_data:
                batch = Batch(*batch_data)
                pred = self(batch.X)
                loss = self.loss(pred, batch.cp, batch.result)
                running_loss += loss.item()
                n_batches += 1
            val_loss = running_loss / n_batches
        self.train()
        self.scheduler.step(val_loss)
        return val_loss

    def checkpoint(self, epoch):

        torch.save(
            self.state_dict(), config.SAVE_PATH / f"model_state_dict_{epoch}.pth"
        )

        param_dict = {
            name: torch.flatten(
                tensor.T if "input" in name or "residual" in name else tensor
            )
            for name, tensor in self.state_dict().items()
        }

        with open(config.SAVE_PATH / f"model_parameters_{epoch}.rs", "w") as file:
            for name, param in param_dict.items():
                if name == "l1.bias":
                    param = param.mul(config.SCALE)
                param = param.mul(config.SCALE).round().to(torch.int16)
                file.write(
                    f"pub const {name.replace('.', '_').upper()}: [i16; {len(param)}] = {param.tolist()};\n\n"
                )


if __name__ == "__main__":
    import os
    from fen_parser import fen_to_vec

    model = NNUE()
    model.load_state_dict(
        torch.load("./nets/2024-03-01T20-57-31/model_state_dict_32.pth")
    )
    print(
        600
        * model(
            torch.Tensor(fen_to_vec("2r5/8/4k3/4p1p1/1KP1P3/2N2P1p/1P6/8 w - - 0 56"))
            .reshape(1, -1)
            .to(config.DEVICE)
        )
    )
    os.mkdir(config.SAVE_PATH)
    model.checkpoint(999)
