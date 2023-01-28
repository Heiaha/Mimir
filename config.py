import datetime
import torch

from pathlib import Path

############################################
# Hyperpatameters
############################################
INPUT_FEATURES = 768
L1 = 256
CP_SCALING = 0.0045235127
N_BUCKETS = 8
LAMBDA = 0.8

############################################
# Training
############################################
BATCH_SIZE = 8192
LEARNING_RATE = 1e-3
N_WORKERS = 8
VALIDATION_CHECKS_PER_EPOCH = 10
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


############################################
# Paths
############################################
SAVE_PATH = Path(f"nets/{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}/")
TRAINING_DIR = Path("training/")
VALIDATION_DIR = Path("testing/")
