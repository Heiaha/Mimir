import datetime
import os
import torch
import torch.optim as optim
import yaml

import losses
from model import NNUE
from dataset import PositionVectorIterableDataset, Batch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from glob import glob
from tqdm import tqdm
from math import isclose


def train_loop(model, dataloader, loss_fn, optimizer, device):
    model.train()

    for batch_data in tqdm(dataloader):
        batch = Batch(*batch_data).to(device)

        optimizer.zero_grad()
        pred = model(batch.x)
        loss = loss_fn(pred, batch.cp, batch.result)

        # Backpropagation
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if "psqt" not in name:
                    param.clamp_(-1.98, 1.98)


@torch.no_grad()
def test_loop(model, dataloader, loss_fn, device):
    model.eval()
    n_batches = len(dataloader)

    loss = 0
    for batch_data in dataloader:
        batch = Batch(*batch_data).to(device)
        pred = model(batch.x)
        loss += loss_fn(pred, batch.cp, batch.result)

    return loss / n_batches


def main(config_filename):

    with open(config_filename) as file:
        config = yaml.safe_load(file)

    print(f"Running on cuda: {torch.cuda.is_available()}")
    training_files = glob(f"{config['training']['training_dir']}/*")
    testing_files = glob(f"{config['training']['testing_dir']}/*")
    save_path = os.path.join(
        config["training"]["save_dir"],
        datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H-%M-%S"),
    )
    os.mkdir(save_path)

    training_dataset = PositionVectorIterableDataset(training_files)
    validation_dataset = PositionVectorIterableDataset(testing_files)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=min(len(training_files), config["training"]["workers"]),
        drop_last=True,
    )

    testing_dataloader = DataLoader(
        validation_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=min(len(testing_files), config["training"]["workers"]),
        drop_last=True,
    )

    model = NNUE(**config["model"]).to(config["training"]["device"])
    loss_fn = losses.ScaledMSELoss(**config["scaling"])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=0, threshold=0)

    best_test_loss = float("inf")

    for epoch in range(100):

        train_loop(
            model, training_dataloader, loss_fn, optimizer, config["training"]["device"]
        )
        test_loss = test_loop(
            model, testing_dataloader, loss_fn, config["training"]["device"]
        )
        scheduler.step(test_loss)
        lr, *_ = scheduler.get_last_lr()

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": test_loss,
            },
            os.path.join(save_path, f"model_checkpoint_{epoch}.pth"),
        )

        model.save_quantized(epoch, save_path)

        print(f"Epoch {epoch} validation loss: {test_loss}")
        print(f"New lr: {lr}")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
        elif isclose(lr, 1e-6):
            print("Finished Training")
            return


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main("config.yml")
