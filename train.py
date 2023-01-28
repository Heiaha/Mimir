import config
import os
import torch

import itertools
from model import NNUE
from dataset import PositionVectorDataset, Batch
from torch.utils.data import DataLoader
from glob import glob


def main():
    print(f"Running on: {config.DEVICE}")
    training_files = glob(f"{config.TRAINING_DIR}/*")
    validation_files = glob(f"{config.VALIDATION_DIR}/*")
    os.mkdir(config.SAVE_PATH)

    model = NNUE()

    training_dataset = PositionVectorDataset(training_files)
    validation_dataset = PositionVectorDataset(validation_files)

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.N_WORKERS,
        drop_last=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.N_WORKERS,
        drop_last=True,
    )

    total_batches = len(training_dataset) // config.BATCH_SIZE
    batch_check_n = total_batches // config.VALIDATION_CHECKS_PER_EPOCH

    best_val_loss = float("inf")
    print(f"Batches in epoch: {total_batches}")

    for epoch in itertools.count(1):
        running_loss = 0
        for batch_n, batch_tensors in enumerate(training_dataloader, start=1):
            batch = Batch(*batch_tensors)

            # forward + backward + optimize
            loss = model.step(batch)
            running_loss += loss.item()

            if batch_n % batch_check_n == 0:
                training_loss = running_loss / batch_check_n
                running_loss = 0.0

                val_loss = model.val_loss(validation_dataloader)
                print(
                    f"Running Loss: {training_loss:.6f}, Validation Loss: {val_loss:.6f}"
                )
        val_loss = model.val_loss(validation_dataloader)
        if val_loss > best_val_loss:
            print("Finished Training")
            return
        best_val_loss = val_loss
        print(f"Epoch {epoch} validation loss: {val_loss}")
        model.checkpoint(epoch)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
