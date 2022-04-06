import os
import datetime
import torch
import torch.optim as optim
import json
import utils
import itertools
from model import NNUE
from dataset import PositionVectorDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from glob import glob


BATCH_SIZE = 8192
LEARNING_RATE = 1e-3
N_WORKERS = 8
VALIDATION_CHECKS_PER_EPOCH = 10
SAVE_PATH = f"nets/{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


TRAINING_DIR = "training/"
TESTING_DIR = "testing/"


def step(model, optimizer, X, score, result):
    pred = model(X)
    loss = model.loss(pred, score, result)
    loss.backward()
    optimizer.step()
    # zero the parameter gradients
    optimizer.zero_grad()
    return loss


def calc_val_loss(model, val_data):
    model.eval()
    with torch.no_grad():
        running_loss = 0
        n_batches = 0
        for X, score, result in val_data:
            X, score, result = X.to(DEVICE), score.to(DEVICE), result.to(DEVICE)
            pred = model(X)
            loss = model.loss(pred, score, result)
            running_loss += loss.item()
            n_batches += 1
        val_loss = running_loss / n_batches
    model.train()
    return val_loss


def checkpoint(model, epoch):
    torch.save(model.state_dict(), SAVE_PATH + f"/model_state_dict_{epoch}.pth")
    save_dict = {
        name: torch.flatten(
            tensor.T if "input" in name or "psqt" in name else tensor
        ).tolist()
        for name, tensor in model.state_dict().items()
    }

    with open(SAVE_PATH + f"/model_parameters_raw_{epoch}.txt", "w") as file:
        file.write(json.dumps(save_dict))

    save_dict = utils.quantize(save_dict)

    with open(SAVE_PATH + f"/model_parameters_{epoch}.txt", "w") as file:
        file.write(json.dumps(save_dict))


def main():
    print(f"Running on: {DEVICE}")
    training_files = glob(f"{TRAINING_DIR}/*")
    validation_files = glob(f"{TESTING_DIR}/*")
    os.mkdir(SAVE_PATH)

    model = NNUE().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", patience=1, verbose=True, min_lr=1e-8
    )

    training_dataloader = DataLoader(
        PositionVectorDataset(training_files),
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
        drop_last=True,
    )

    validation_dataloader = DataLoader(
        PositionVectorDataset(validation_files),
        batch_size=BATCH_SIZE,
        num_workers=N_WORKERS,
        drop_last=True,
    )

    total_batches = utils.count_fens(TRAINING_DIR) // BATCH_SIZE
    batch_check_n = total_batches // VALIDATION_CHECKS_PER_EPOCH

    best_val_loss = float("inf")
    print(f"Batches in epoch: {total_batches}")

    for epoch in itertools.count(1):
        epoch_batches = 0
        running_loss = 0
        for X, score, result in training_dataloader:
            epoch_batches += 1

            X, score, result = X.to(DEVICE), score.to(DEVICE), result.to(DEVICE)

            # forward + backward + optimize
            loss = step(model, optimizer, X, score, result)
            running_loss += loss.item()

            if epoch_batches % batch_check_n == 0:
                training_loss = running_loss / batch_check_n
                running_loss = 0.0

                val_loss = calc_val_loss(model, validation_dataloader)
                scheduler.step(val_loss)
                print(
                    f"Running Loss: {training_loss:.6f}, Validation Loss: {val_loss:.6f}"
                )

        val_loss = calc_val_loss(model, validation_dataloader)
        if val_loss > best_val_loss:
            break
        best_val_loss = val_loss
        print(f"Epoch {epoch} validation loss: {val_loss}")

        checkpoint(model, epoch)

    print("Finished Training")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
