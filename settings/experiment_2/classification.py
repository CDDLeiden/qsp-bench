from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import deepchem as dc
from xgboost import XGBClassifier

from qsprpred import TargetProperty, TargetTasks
from qsprpred.benchmarks import BenchmarkSettings
from qsprpred.extra.gpu.models.dnn import DNNModel
from qsprpred.extra.gpu.models.neural_network import STFullyConnected
from qsprpred.extra.models.random import RandomModel, RatioDistributionAlgorithm
from qsprpred.models import SklearnModel, TestSetAssessor
from qsprpred.models.monitors import NullMonitor
from settings.base import *
from utils.data_sources import MoleculeNetSource

# MoleculeNet dataset settings
# Add your own specs to the dictionary below
sets_to_loads = {
    "HIV": {
        "loader": dc.molnet.load_hiv,
        "smiles_col": "smiles",
    },
    "bace": {
        "loader": dc.molnet.load_bace_classification,
        "smiles_col": "mol",
    },
    "BBBP": {
        "loader": dc.molnet.load_bbbp,
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
    [TargetProperty(prop_name, TargetTasks.SINGLECLASS, th="precomputed")],
]

# models
MODELS = [
    SklearnModel(
        name=f"{NAME}_XGBClassifier",
        alg=XGBClassifier,
        base_dir=MODELS_DIR,
        parameters={"n_jobs": 1},
    ),
    RandomModel(
        name=f"{NAME}_RatioModel",
        base_dir=MODELS_DIR,
        alg=RatioDistributionAlgorithm,
    ),
]

# assessors
ASSESSORS = [
    TestSetAssessor(scoring="roc_auc", monitor=NullMonitor(), use_proba=True),
    TestSetAssessor(
        scoring="matthews_corrcoef", monitor=NullMonitor(), use_proba=False
    ),
    TestSetAssessor(scoring="recall", monitor=NullMonitor(), use_proba=False),
    TestSetAssessor(scoring="precision", monitor=NullMonitor(), use_proba=False),
    TestSetAssessor(scoring="f1", monitor=NullMonitor(), use_proba=False),
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
