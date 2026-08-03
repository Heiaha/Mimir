import datetime
import torch
import torch.optim as optim
import yaml

import losses
from model import NNUE
from dataset import PositionVectorIterableDataset, count_features
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from tqdm import tqdm

# Reading a running loss syncs the device, so do it rarely.
LOG_EVERY = 100


def train_loop(model, dataloader, loss_fn, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = torch.zeros((), device=device)
    params = list(model.parameters())
    # The engine's fused kernel multiplies a folded (real + virtual) hidden
    # weight by a 255-clamped activation in i16, so each half must stay
    # within half the i16-safe range.
    hidden_weights = [model.hidden_layer.weight, model.virtual_hidden_layer.weight]
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=False)
    for idx, batch in enumerate(pbar):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        pred = model(batch)
        loss = loss_fn(pred, batch)

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.detach()

        if idx % LOG_EVERY == 0:
            pbar.set_postfix(
                {
                    "loss": total_loss.item() / (idx + 1),
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        with torch.no_grad():
            torch._foreach_clamp_min_(params, -1.98)
            torch._foreach_clamp_max_(params, 1.98)
            torch._foreach_clamp_min_(hidden_weights, -0.99)
            torch._foreach_clamp_max_(hidden_weights, 0.99)


def wdl_train_loop(model, dataloader, loss_fn, optimizer, device, epoch):
    model.eval()  # the trunk is frozen; only the WDL head trains
    total_loss = torch.zeros((), device=device)
    pbar = tqdm(dataloader, desc=f"WDL Training Epoch {epoch}", leave=False)
    for idx, batch in enumerate(pbar):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        loss = loss_fn(model.forward_wdl(batch), batch)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.detach()
        if idx % LOG_EVERY == 0:
            pbar.set_postfix({"loss": total_loss.item() / (idx + 1)})


@torch.no_grad()
def wdl_test_loop(model, dataloader, loss_fn, device, epoch):
    model.eval()
    pbar = tqdm(dataloader, desc=f"WDL Testing Epoch {epoch}", leave=True)
    loss = torch.zeros((), device=device)
    predicted = torch.zeros(3, device=device)
    actual = torch.zeros(3, device=device)
    n_rows = 0
    for idx, batch in enumerate(pbar):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        logits = model.forward_wdl(batch)
        loss += loss_fn(logits, batch).detach()

        predicted += logits.softmax(-1).sum(dim=0)
        actual += torch.bincount((batch["result"].flatten() * 2).long(), minlength=3)
        n_rows += logits.shape[0]

        if idx % LOG_EVERY == 0:
            pbar.set_postfix({"loss": loss.item() / (idx + 1)})

    return (
        loss.item() / len(dataloader),
        (predicted / n_rows).cpu(),
        (actual / n_rows).cpu(),
    )


@torch.no_grad()
def test_loop(model, dataloader, loss_fn, device, epoch):
    model.eval()
    pbar = tqdm(dataloader, desc=f"Testing Epoch {epoch}", leave=True)
    loss = torch.zeros((), device=device)
    for idx, batch in enumerate(pbar):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        pred = model(batch)
        loss += loss_fn(pred, batch).detach()
        if idx % LOG_EVERY == 0:
            pbar.set_postfix({"loss": loss.item() / (idx + 1)})

    return loss.item() / len(dataloader)


def main(config_filename):
    torch.manual_seed(0)

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
        pin_memory=device == "cuda",
    )

    testing_dataloader = DataLoader(
        testing_dataset,
        batch_size=None,
        pin_memory=device == "cuda",
        num_workers=min(len(testing_files), config["training"]["workers"]),
    )

    counts, n_pos = count_features(training_dataloader)
    counts_per_batch = counts.float() * (batch_size / n_pos)
    print(f"Feature counts: {n_pos:,} positions, {int((counts > 0).sum())} features seen")

    model = NNUE(counts_per_batch=counts_per_batch, **config["model"]).to(device)

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    loss_fn = losses.ScaledPowLoss(**config["scaling"])

    # The WDL head trains in its own phase, on a frozen trunk.
    eval_params = [
        param
        for name, param in model.named_parameters()
        if not name.startswith("wdl_layer")
    ]
    optimizer = optim.Adam(eval_params, lr=config["training"]["max_lr"], fused=True)

    # An upper bound is required: per-worker drop_last only ever yields fewer.
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
        # The WDL head is the only key permitted to be absent.
        missing, unexpected = model.load_state_dict(
            checkpoint_info["model_state_dict"], strict=False
        )
        assert not unexpected and all(k.startswith("wdl_layer.") for k in missing)
        optimizer.load_state_dict(checkpoint_info["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_info["scheduler_state_dict"])
        starting_epoch = checkpoint_info["epoch"] + 1
        best_loss = checkpoint_info.get("best_loss", float("inf"))

    save_path.mkdir(parents=True, exist_ok=True)

    model.save_quantized(-1, save_path)

    for epoch in range(starting_epoch, epochs + 1):
        # Worker seeds derive from the torch RNG, so this fixes the shuffle
        # order per epoch and makes stop/resume match an uninterrupted run.
        torch.manual_seed(epoch)
        train_loop(
            model, training_dataloader, loss_fn, optimizer, scheduler, device, epoch
        )
        test_loss = test_loop(model, testing_dataloader, loss_fn, device, epoch)
        lr, *_ = scheduler.get_last_lr()

        improved = test_loss < best_loss
        if improved:
            best_loss = test_loss
            model.save_quantized("best", save_path)

        print(
            f"Epoch {epoch}: test_loss={test_loss:.4e} lr={lr:.2e}"
            f" best={best_loss:.4e}{' *' if improved else ''}"
        )

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

    wdl_loss_fn = losses.WDLLoss()
    wdl_optimizer = optim.Adam(
        model.wdl_layer.parameters(), lr=config["training"]["max_lr"], fused=True
    )

    for epoch in range(1, config["training"]["wdl_epochs"] + 1):
        wdl_train_loop(
            model, training_dataloader, wdl_loss_fn, wdl_optimizer, device, epoch
        )
        test_ce, predicted, actual = wdl_test_loop(
            model, testing_dataloader, wdl_loss_fn, device, epoch
        )
        print(
            f"WDL Epoch {epoch}: test_ce={test_ce:.4e}"
            f" predicted L/D/W={predicted[0]:.3f}/{predicted[1]:.3f}/{predicted[2]:.3f}"
            f" actual={actual[0]:.3f}/{actual[1]:.3f}/{actual[2]:.3f}"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": test_ce,
            },
            save_path / "model_checkpoint_wdl.pth",
        )

        model.save_quantized("wdl", save_path)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main("config.yml")
