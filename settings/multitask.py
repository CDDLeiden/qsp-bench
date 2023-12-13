from sklearn.neighbors import KNeighborsRegressor

from qsprpred import TargetProperty, TargetTasks
from qsprpred.benchmarks import BenchmarkSettings
from qsprpred.extra.gpu.models.pyboost import PyBoostModel
from qsprpred.models import TestSetAssessor, SklearnModel
from .data_sources import PapyrusForBenchmark
from .base import *

# follow tutorial: https://gitlab.services.universiteitleiden.nl/cdd/QSPRpred/-/blob/dev/tutorials/advanced/modelling/multi_task_modelling.ipynb?ref_type=heads
# we do regression only because PyBoostModel does not support classification

# data sources
# TODO: create one multitask dataset
DATA_SOURCES = [
    PapyrusForBenchmark(["P30542"], f"{DATA_DIR}/sets", n_samples=100),  # A1
    PapyrusForBenchmark(["P29274"], f"{DATA_DIR}/sets", n_samples=100),  # A2A
    PapyrusForBenchmark(["P29275"], f"{DATA_DIR}/sets", n_samples=100),  # A2B
    PapyrusForBenchmark(["P0DMS8"], f"{DATA_DIR}/sets", n_samples=100),  # A3
]

MODELS = [
        # TODO: add random model
        PyBoostModel(
            name=f"{NAME}_PyBoost",
            base_dir=MODELS_DIR,
        ),
        SklearnModel(
            alg=KNeighborsRegressor,
            name=f"{NAME}_KNeighborsRegressor",
            base_dir=MODELS_DIR,
        )
    ]

# target properties
# TODO: needs  to be  converted to multitask too
TARGET_PROPS = [
    [TargetProperty("pchembl_value_Median", TargetTasks.REGRESSION, th=[7])],
]

# assessors
ASSESSORS = [
    TestSetAssessor(scoring="roc_auc"),
    TestSetAssessor(scoring="matthews_corrcoef", use_proba=False)
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
