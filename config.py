import datetime
import torch

from pathlib import Path

############################################
# Hyperpatameters
############################################
L0 = 768
L1 = 256

LAMBDA = 1.0
PATIENCE = 1

############################################
# Training
############################################
BATCH_SIZE = 8192
LEARNING_RATE = 1e-3
N_WORKERS = 8
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


############################################
# Paths
############################################
SAVE_PATH = Path(f"nets/{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}/")
TRAINING_DIR = Path("training/")
VALIDATION_DIR = Path("testing/")

############################################
# Scaling
############################################
NNUE_2_SCORE = 600
CP_SCALING = 400
SCALE = 64
