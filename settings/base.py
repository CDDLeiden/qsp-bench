import os
import shutil

from qsprpred.benchmarks import DataPrepSettings
from qsprpred.data import RandomSplit, ClusterSplit
from qsprpred.data.descriptors.sets import FingerprintSet

START_FRESH = False  # set True to run all replicas from scratch
RESET_MODELS = False  # set True to reset all models
NAME = "ExampleBenchmark"  # name of the benchmark
SEED = 42  # random seed
N_REPLICAS = 15  # number of repetitions per experiment
DATA_DIR = f"./data/{NAME}"  # directory to store data
MODELS_DIR = f"{DATA_DIR}/models"  # directory to store models
N_PROC = 12  # number of processes to use for parallelization
RESULTS_FILE = f"{DATA_DIR}/results.tsv"  # file to store results

# descriptors
DESCRIPTORS = [
    [FingerprintSet("MorganFP", radius=3, nBits=2048)],
]

# data preparation settings
DATA_PREPS = [
    DataPrepSettings(
        split=RandomSplit(test_fraction=0.2),
    ),
    DataPrepSettings(
        split=ClusterSplit(test_fraction=0.2),
    ),
]

# create data directory
if os.path.exists(DATA_DIR) and START_FRESH:
    shutil.rmtree(DATA_DIR)
if os.path.exists(MODELS_DIR) and RESET_MODELS:
    shutil.rmtree(MODELS_DIR)
if os.path.exists(RESULTS_FILE) and RESET_MODELS:
    os.remove(RESULTS_FILE)
