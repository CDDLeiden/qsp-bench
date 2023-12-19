from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor

from utils.settings.base import *
from utils.settings.data_sources import PapyrusForBenchmark
from qsprpred import TargetProperty
from qsprpred.models import SklearnModel, TestSetAssessor
from qsprpred.benchmarks import BenchmarkSettings
from qsprpred.extra.gpu.models.pyboost import PyBoostModel

# data sources
DATA_SOURCES = [
    PapyrusForBenchmark(
        ["P30542"],
        f"{DATA_DIR}/sets", n_samples=N_SAMPLES
    ),
    PapyrusForBenchmark(
        ["P29274"],
        f"{DATA_DIR}/sets", n_samples=N_SAMPLES
    ),
    PapyrusForBenchmark(
        ["P29275"],
        f"{DATA_DIR}/sets", n_samples=N_SAMPLES
    ),
    PapyrusForBenchmark(
        ["P0DMS8"],
        f"{DATA_DIR}/sets", n_samples=N_SAMPLES
    ),
]

MODELS = [
    PyBoostModel(
        name=f"{NAME}_PyBoost",
        base_dir=MODELS_DIR,
        parameters={
            "loss": "mse",
            "metric": "r2_score",
            "ntrees": 1000,
        }
    ),
    SklearnModel(
        alg=KNeighborsRegressor,
        name=f"{NAME}_KNeighborsRegressor",
        base_dir=MODELS_DIR,
    )
]

TARGET_PROPS = [
    [TargetProperty.fromDict({
        "name": "pchembl_value_Mean",
        "task": "REGRESSION",
        "imputer": SimpleImputer(strategy="median")
    })],
]

ASSESSORS = [
    TestSetAssessor(
        scoring="r2",
        split_multitask_scores=True
    ),
    TestSetAssessor(
        scoring="neg_mean_squared_error",
        use_proba=False,
        split_multitask_scores=True
    ),
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
