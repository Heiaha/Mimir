import config
import os
import torch

import itertools
from model import NNUE
from dataset import PositionVectorDataset, Batch
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm


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
        num_workers=min(
            len(glob(f"{config.TRAINING_DIR}/*")),
            config.N_WORKERS
        ),
        drop_last=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=min(
            len(glob(f"{config.VALIDATION_DIR}/*")),
            config.N_WORKERS
        ),
        drop_last=True,
    )

    total_batches = len(training_dataset) // config.BATCH_SIZE

    print(f"Batches in epoch: {total_batches}")

    epochs_no_improve = 0
    best_val_loss = float("inf")

    for epoch in itertools.count():
        for batch_tensors in tqdm(
                training_dataloader,
                total=total_batches
        ):
            batch = Batch(*batch_tensors)

            # forward + backward + optimize
            model.step(batch)

        val_loss = model.val_loss(validation_dataloader)
        model.checkpoint(epoch)

        print(f"Epoch {epoch} validation loss: {val_loss}")

        if val_loss < best_val_loss:
            epochs_no_improve = 0
            best_val_loss = val_loss
        else:
            epochs_no_improve += 1

        if epochs_no_improve > config.PATIENCE:
            print("Finished Training")
            return


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
