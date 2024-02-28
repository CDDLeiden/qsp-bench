from sklearn.metrics import root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

import deepchem as dc
from xgboost import XGBRegressor

from qsprpred import TargetProperty, TargetTasks
from qsprpred.benchmarks import BenchmarkSettings
from qsprpred.extra.gpu.models.dnn import DNNModel
from qsprpred.extra.gpu.models.neural_network import STFullyConnected
from qsprpred.extra.models.random import RandomModel, MedianDistributionAlgorithm
from qsprpred.models import SklearnModel, TestSetAssessor
from qsprpred.models.monitors import NullMonitor
from settings.base import *
from utils.data_sources import MoleculeNetSource


# some data sets might need unzipping first
def unzip_with_gzip(file):
    """Just unzip a file with gzip."""
    import gzip
    import shutil
    import os

    with gzip.open(file, "rb") as f_in:
        new_file = file.replace(".gz", "")
        with open(new_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(file)
    return new_file


# MoleculeNet dataset settings
# Add your own specs to the dictionary below
sets_to_loads = {
    "Lipophilicity": {
        "loader": dc.molnet.load_lipo,
        "smiles_col": "smiles",
    },
    "delaney-processed": {
        "loader": dc.molnet.load_delaney,
        "smiles_col": "smiles",
    },
    "freesolv": {
        "loader": dc.molnet.load_freesolv,
        "smiles_col": "smiles",
    },
    "clearance": {
        "loader": dc.molnet.load_clearance,
        "smiles_col": "smiles",
    },
}
if not os.environ.get("QSPBENCH_MOLNETSET"):
    raise ValueError(
        f"Please set the QSPBENCH_MOLNETSET environment variable. "
        f"Options are: {list(sets_to_loads.keys())}"
    )
DATASET = os.environ["QSPBENCH_MOLNETSET"]
prop_name = sets_to_loads[DATASET]["loader"](data_dir=DATA_DIR, save_dir=DATA_DIR)[0][
    0
]  # get the first property only
RESULTS_FILE = RESULTS_FILE.replace(".tsv", f"_{DATASET}_{prop_name}.tsv")
if DATASET == "freesolv" and not os.path.exists(f"{DATA_DIR}/freesolv.csv"):
    # unzip the file
    unzip_with_gzip(f"{DATA_DIR}/freesolv.csv.gz")

# data sources
DATA_SOURCES = [
    MoleculeNetSource(
        DATASET,
        DATA_DIR,
        n_samples=N_SAMPLES,
        smiles_col=sets_to_loads[DATASET]["smiles_col"],
    ),
]

# target properties
TARGET_PROPS = [
    [TargetProperty(prop_name, TargetTasks.REGRESSION)],
]

# models
MODELS = [
    SklearnModel(
        name=f"{NAME}_XGBRegressor",
        alg=XGBRegressor,
        base_dir=MODELS_DIR,
        parameters={"n_jobs": 1},
    ),
    RandomModel(
        name=f"{NAME}_MedianModel", base_dir=MODELS_DIR, alg=MedianDistributionAlgorithm
    ),
]

# assessors
ASSESSORS = [
    TestSetAssessor(scoring="r2", monitor=NullMonitor()),
    TestSetAssessor(root_mean_squared_error, monitor=NullMonitor()),
]

# benchmark settings
SETTINGS = BenchmarkSettings(
    name=NAME,
    n_replicas=N_REPLICAS,
    random_seed=SEED,
    data_sources=DATA_SOURCES,
    descriptors=DESCRIPTORS,
    target_props=TARGET_PROPS,
    prep_settings=DATA_PREPS,
    models=MODELS,
    assessors=ASSESSORS,
)
