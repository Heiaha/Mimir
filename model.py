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
        self.input_layer = nn.Linear(config.INPUT_FEATURES, config.L1)
        self.hidden_layer = nn.Linear(config.L1, config.N_BUCKETS)

        self.optimizer = optim.Adam(self.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", patience=1, verbose=True, min_lr=1e-8
        )
        self.to(config.DEVICE)

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
    def loss(pred, score, result):
        wdl_eval_model = (pred * config.CP_SCALING).sigmoid()
        wdl_eval_target = (score * config.CP_SCALING).sigmoid()

        wdl_value_target = config.LAMBDA * wdl_eval_target + (1 - config.LAMBDA) * result

        return (wdl_eval_model - wdl_value_target).square().mean()

    def step(self, batch):
        pred = self(batch.X)
        loss = self.loss(pred, batch.score, batch.result)
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
                loss = self.loss(pred, batch.score, batch.result)
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

        quantized_weights = self.quantize()

        with open(config.SAVE_PATH / f"model_parameters_{epoch}.txt", "w") as file:
            file.write(
                f"pub const NNUE_INPUT_WEIGHTS: [i16; {len(quantized_weights['input_layer.weight'])}] = {quantized_weights['input_layer.weight']};"
            )

            file.write(
                f"pub const NNUE_INPUT_BIASES: [i16; {len(quantized_weights['input_layer.bias'])}] = {quantized_weights['input_layer.bias']};"
            )

            file.write(
                f"pub const NNUE_HIDDEN_WEIGHTS: [i16; {len(quantized_weights['hidden_layer.weight'])}] = {quantized_weights['hidden_layer.weight']};"
            )

            file.write(
                f"pub const NNUE_HIDDEN_BIASES: [i16; {len(quantized_weights['hidden_layer.bias'])}] = {quantized_weights['hidden_layer.bias']};"
            )

    def quantize(self):
        new_dict = defaultdict(list)

        param_dict = {
            name: torch.flatten(
                tensor.T if "input" in name or "psqt" in name else tensor
            ).tolist()
            for name, tensor in self.state_dict().items()
        }

        for name, weights in param_dict.items():
            for value in weights:
                new_dict[name].append(round(value * 64))

        return dict(new_dict)
