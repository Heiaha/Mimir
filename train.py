import datetime
import os
import torch
import torch.optim as optim
import yaml

import losses
from model import NNUE
from dataset import PositionVectorIterableDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from glob import glob
from tqdm import tqdm

EPOCHS = 100

def train_loop(model, dataloader, loss_fn, optimizer, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=False)
    for idx, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}

        pred = model(batch)
        loss = loss_fn(pred, batch)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        pbar.set_postfix({
            "loss": total_loss / (idx + 1),
            "lr": optimizer.param_groups[0]["lr"],
        })

        with torch.no_grad():
            for name, param in model.named_parameters():
                param.clamp_(-1.98, 1.98)


@torch.no_grad()
def test_loop(model, dataloader, loss_fn, device, epoch):
    model.eval()
    pbar = tqdm(dataloader, desc=f"Testing Epoch {epoch}", leave=True)
    loss = 0
    for idx, batch in enumerate(pbar):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch)
        loss += loss_fn(pred, batch).item()
        pbar.set_postfix({"loss": loss / (idx + 1)})

    return loss / len(dataloader)


def main(config_filename):

    with open(config_filename) as file:
        config = yaml.safe_load(file)

    print(f"Running on cuda: {torch.cuda.is_available()}")
    training_files = glob(f"{config['training']['training_dir']}/*")
    testing_files = glob(f"{config['training']['testing_dir']}/*")

    checkpoint_path = config.get("load")

    save_path = os.path.dirname(checkpoint_path) if checkpoint_path else os.path.join(
        config["training"]["save_dir"],
        datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H-%M-%S"),
    )
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    training_dataset = PositionVectorIterableDataset(training_files)
    testing_dataset = PositionVectorIterableDataset(testing_files)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=min(len(training_files), config["training"]["workers"]),
        drop_last=True,
    )

    testing_dataloader = DataLoader(
        testing_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=min(len(testing_files), config["training"]["workers"]),
        drop_last=True,
    )

    model = NNUE(**config["model"]).to(config["training"]["device"])
    loss_fn = losses.ScaledMSELoss(**config["scaling"])
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    starting_epoch = 0

    if checkpoint_path:
        checkpoint_info = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_info["model_state_dict"])
        optimizer.load_state_dict(checkpoint_info["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_info["scheduler_state_dict"])
        starting_epoch = checkpoint_info["epoch"] + 1

    for epoch in range(starting_epoch, EPOCHS):

        train_loop(
            model, training_dataloader, loss_fn, optimizer, config["training"]["device"], epoch
        )
        test_loss = test_loop(
            model, testing_dataloader, loss_fn, config["training"]["device"], epoch
        )
        scheduler.step()
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


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main("config.yml")
