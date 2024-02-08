from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from settings.base import *
from utils.data_sources import PapyrusForBenchmark
from qsprpred import TargetProperty
from qsprpred.models import TestSetAssessor, SklearnModel
from qsprpred.benchmarks import BenchmarkSettings

from qsprpred.extra.models.random import (
    RandomModel,
    MedianDistributionAlgorithm,
    ScipyDistributionAlgorithm,
)

# data sources
DATA_SOURCES = [
    PapyrusForBenchmark(["P30542"], f"{DATA_DIR}/sets", n_samples=N_SAMPLES),
    PapyrusForBenchmark(["P29274"], f"{DATA_DIR}/sets", n_samples=N_SAMPLES),
    PapyrusForBenchmark(["P29275"], f"{DATA_DIR}/sets", n_samples=N_SAMPLES),
    PapyrusForBenchmark(["P0DMS8"], f"{DATA_DIR}/sets", n_samples=N_SAMPLES),
]

MODELS = [
    RandomModel(
        name=f"{NAME}_Random", base_dir=MODELS_DIR, alg=MedianDistributionAlgorithm
    ),
    SklearnModel(
        alg=KNeighborsRegressor,
        name=f"{NAME}_KNeighborsRegressor",
        base_dir=MODELS_DIR,
    ),
    SklearnModel(
        alg=XGBRegressor,
        name=f"{NAME}_XGBRegressor_MOT",
        base_dir=MODELS_DIR,
        parameters={
            "n_jobs": 1,
        },
    ),
]

TARGET_PROPS = [
    [
        TargetProperty.fromDict(
            {
                "name": "pchembl_value_Mean",
                "task": "REGRESSION",
                "imputer": SimpleImputer(strategy="median"),
            }
        )
    ],
]

ASSESSORS = [
    TestSetAssessor(scoring="r2"),
    TestSetAssessor(
        scoring=root_mean_squared_error,
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
