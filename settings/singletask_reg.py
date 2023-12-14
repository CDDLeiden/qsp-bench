from .base import *
from .data_sources import PapyrusForBenchmark
from qsprpred import TargetProperty, TargetTasks
from qsprpred.models import SklearnModel, TestSetAssessor
from qsprpred.benchmarks import BenchmarkSettings
from xgboost import XGBRegressor
from qsprpred.extra.gpu.models.pyboost import PyBoostModel

# data sources
DATA_SOURCES = [
    PapyrusForBenchmark(
        ["P30542", "P29274", "P29275", "P0DMS8"],
        f"{DATA_DIR}/sets", n_samples=100
    ),
]

MODELS=[
    SklearnModel(
        name="XGBR",
        alg=XGBRegressor,
        base_dir=f"{MODELS_DIR}/models",
    ),
    PyBoostModel(
        name=f"{NAME}_PyBoost",
        base_dir=MODELS_DIR,
        parameters={
            "loss": "mse",
            "metric": "r2_score"
        }
    ),
]

TARGET_PROPS=[
    # one or more properties to model
    [
        TargetProperty.fromDict(
            {
                "name": "pchembl_value_Mean",
                "task": TargetTasks.REGRESSION,
            }
        )
    ],
]

ASSESSORS=[
    TestSetAssessor(scoring="r2"),
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
