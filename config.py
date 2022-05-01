import datetime
import torch

############################################
# Hyperpatameters
############################################
INPUT_FEATURES = 768
L1 = 256
CP_SCALING = 0.007828325269999983
N_BUCKETS = 8
LAMBDA = 1.0
EPSILON = 1e-12

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
SAVE_PATH = f"nets/{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}/"
TRAINING_DIR = "training/"
TESTING_DIR = "testing/"