import datetime
import torch
import torch.optim as optim
import yaml

import losses
from model import NNUE
from dataset import PositionVectorIterableDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from tqdm import tqdm


def train_loop(model, dataloader, loss_fn, optimizer, scheduler, device, epoch):
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
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()

        pbar.set_postfix({
            "loss": total_loss / (idx + 1),
            "lr": optimizer.param_groups[0]["lr"],
        })

        with torch.no_grad():
            for param in model.parameters():
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
    training_files = sorted(Path(config["training"]["training_dir"]).glob("*"))
    testing_files = sorted(Path(config["training"]["testing_dir"]).glob("*"))

    checkpoint_path = config.get("load")

    save_path = (
        Path(checkpoint_path).parent
        if checkpoint_path
        else Path(config["training"]["save_dir"])
        / datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H-%M-%S")
    )

    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    device = config["training"]["device"]

    training_dataset = PositionVectorIterableDataset(training_files, batch_size)
    testing_dataset = PositionVectorIterableDataset(testing_files, batch_size)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=None,  # the dataset yields whole batches
        num_workers=min(len(training_files), config["training"]["workers"]),
    )

    testing_dataloader = DataLoader(
        testing_dataset,
        batch_size=None,
        num_workers=min(len(testing_files), config["training"]["workers"]),
    )

    model = NNUE(**config["model"]).to(device)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    loss_fn = losses.ScaledMSELoss(**config["scaling"])
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["max_lr"])

    # Warmup up to max_lr, then cosine anneal to ~0 over the whole run, so the
    # epoch count is just a budget knob. total_steps must be an upper bound on the
    # optimizer steps -- it is, since per-worker drop_last only yields fewer.
    total_steps = len(training_dataloader) * epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["training"]["max_lr"],
        total_steps=total_steps,
        pct_start=config["training"]["warmup_frac"],
        anneal_strategy="cos",
        cycle_momentum=False,
    )

    starting_epoch = 1
    best_loss = float("inf")

    if checkpoint_path:
        checkpoint_info = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint_info["model_state_dict"])
        optimizer.load_state_dict(checkpoint_info["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_info["scheduler_state_dict"])
        starting_epoch = checkpoint_info["epoch"] + 1
        best_loss = checkpoint_info.get("best_loss", float("inf"))

    save_path.mkdir(parents=True, exist_ok=True)

    model.save_quantized(-1, save_path)

    for epoch in range(starting_epoch, epochs):

        train_loop(
            model, training_dataloader, loss_fn, optimizer, scheduler, device, epoch
        )
        test_loss = test_loop(
            model, testing_dataloader, loss_fn, device, epoch
        )
        lr, *_ = scheduler.get_last_lr()
        print(f"Epoch {epoch}: test_loss={test_loss:.4e} lr={lr:.2e} best={best_loss:.4e}")

        save_path.mkdir(parents=True, exist_ok=True)

        if test_loss < best_loss:
            best_loss = test_loss
            model.save_quantized("best", save_path)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": test_loss,
                "best_loss": best_loss,
            },
            save_path / f"model_checkpoint_{epoch}.pth",
        )

        model.save_quantized(epoch, save_path)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main("config.yml")
