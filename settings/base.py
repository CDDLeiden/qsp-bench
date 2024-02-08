import os
import shutil

from qsprpred.benchmarks import DataPrepSettings
from qsprpred.data import ClusterSplit, RandomSplit
from qsprpred.data.descriptors.fingerprints import MorganFP

START_FRESH = False  # set True to delete all previous outputs
RESET_MODELS = True  # set True to reset all models
N_SAMPLES = None  # samples per target to use for benchmarking, use None for all data
NAME = os.environ["QSPBENCH_SETTINGS"].split(".")[-1]  # name of the benchmark
NAME = NAME + (f"_{N_SAMPLES}" if N_SAMPLES else "")  # append number of samples
SEED = 42  # random seed
N_REPLICAS = 30  # number of repetitions per experiment
DATA_DIR = f"./data/{NAME}"  # directory to store data
MODELS_DIR = f"{DATA_DIR}/models"  # directory to store models
N_PROC = (
    os.cpu_count()
    if "QSPBENCH_NPROC" not in os.environ
    else int(os.environ["QSPBENCH_NPROC"])
)  # number of processes to use
RESULTS_FILE = f"{DATA_DIR}/results.tsv"  # file to store results

# descriptors
DESCRIPTORS = [
    [MorganFP(radius=3, nBits=2048)],
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
