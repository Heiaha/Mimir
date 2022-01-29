import os
import datetime
import torch
import torch.optim as optim
import json
import model
import utils
from dataset import PositionVectorDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from glob import glob


BATCH_SIZE = 8192
LEARNING_RATE = 0.01
MAX_EPOCHS = 200
N_WORKERS = 5
LAMBDA = 0.8
SAVE_PATH = f"nets/lambda{LAMBDA}_lr{LEARNING_RATE}_bs{BATCH_SIZE}_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

TRAINING_DIR = "training/"
TESTING_DIR = "testing/"


def step(net, optimizer, X, score, result):
    pred = net(X)
    loss = model.loss(LAMBDA, pred, score, result)
    loss.backward()
    optimizer.step()
    return loss


def calc_val_loss(net, val_data):
    net.eval()
    with torch.no_grad():
        running_loss = 0
        n_batches = 0
        for X, score, result in val_data:
            X, score, result = (
                X.to(DEVICE),
                score.to(DEVICE),
                result.to(DEVICE),
            )
            pred = net(X)
            loss = model.loss(LAMBDA, pred, score, result)
            running_loss += loss.item()
            n_batches += 1
        val_loss = running_loss/n_batches
    net.train()
    return val_loss


def main():
    training_files = glob(f"{TRAINING_DIR}/*")
    validation_files = glob(f"{TESTING_DIR}/*")

    net = model.NNUE().to(DEVICE)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, patience=1, verbose=True)

    training_dataset = PositionVectorDataset(training_files)
    validation_dataset = PositionVectorDataset(validation_files)

    training_dataloader = DataLoader(
        training_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, drop_last=True
    )

    validation_dataloader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, num_workers=N_WORKERS, drop_last=True
    )

    os.mkdir(SAVE_PATH)
    os.mkdir(SAVE_PATH + "/logs/")
    writer = SummaryWriter(SAVE_PATH + "/logs/")
    total_batches = 0
    for epoch in range(1, MAX_EPOCHS + 1):  # loop over the dataset multiple times
        training_batches = 0
        running_loss = 0
        net.train()
        for i, (X, score, result) in enumerate(training_dataloader, start=1):
            # zero the parameter gradients
            optimizer.zero_grad()

            X, score, result = (
                X.to(DEVICE),
                score.to(DEVICE),
                result.to(DEVICE),
            )
            # forward + backward + optimize
            loss = step(net, optimizer, X, score, result)

            # print statistics
            running_loss += loss.item()
            training_batches += 1
            total_batches += 1
            if training_batches % 10 == 0:  # print every 10 mini-batches
                training_loss = running_loss / 10
                print(f"[{epoch}, {training_batches:5d}] training loss: {training_loss:.6f}")
                writer.add_scalar("training loss", training_loss, total_batches)
                running_loss = 0.0

        print("Finished training.")
        val_loss = calc_val_loss(net, validation_dataloader)
        writer.add_scalar("validation_loss", val_loss, total_batches)
        scheduler.step(val_loss)

        print(f"Validation Loss = {val_loss}")
        print(f"Training Batches = {training_batches}")
        print("\n\n\n")

        torch.save(net.state_dict(), SAVE_PATH + f"/model_state_dict_{epoch}.pth")
        save_dict = {
            name: torch.flatten(tensor.T).tolist()
            for name, tensor in net.state_dict().items()
        }

        save_dict = utils.quantize(save_dict)

        with open(SAVE_PATH + f"/model_parameters_{epoch}.txt", "w") as file:
            file.write(json.dumps(save_dict))

    print("Finished Training")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
