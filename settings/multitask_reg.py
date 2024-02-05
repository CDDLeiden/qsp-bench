from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor

from qsprpred import TargetProperty
from qsprpred.benchmarks import BenchmarkSettings
# from qsprpred.extra.gpu.models.pyboost import PyBoostModel
from qsprpred.models import TestSetAssessor, SklearnModel
from qsprpred.extra.models.random import RandomModel
from utils.settings.data_sources import PapyrusForBenchmarkMT
from utils.settings.base import *

# data sources
ACC_KEYS = ["P30542", "P29274", "P29275", "P0DMS8"]
DATA_SOURCES = [
    PapyrusForBenchmarkMT(
        ACC_KEYS,
        f"{DATA_DIR}/sets", n_samples=N_SAMPLES
    ),
]

MODELS = [
        # PyBoostModel(
        #     name=f"{NAME}_PyBoost",
        #     base_dir=MODELS_DIR,
        #     parameters={
        #         "loss": "mse",
        #         "metric": "r2_score",
        #         "ntrees": 1000,
        #     }
        # ),
        SklearnModel(
            alg=KNeighborsRegressor,
            name=f"{NAME}_KNeighborsRegressor",
            base_dir=MODELS_DIR,
        ),
        RandomModel(
            name=f"{NAME}_RandomModel",
            base_dir=MODELS_DIR,
            alg=ScipyDistributionAlgorithm,
        )
    ]

# target properties
TARGET_PROPS = [
    [
        TargetProperty.fromDict({
            "name": "P0DMS8",
            "task": "REGRESSION",
            "imputer": SimpleImputer(strategy="median")
        }),
        TargetProperty.fromDict({
            "name": "P29274",
            "task": "REGRESSION",
            "imputer": SimpleImputer(strategy="median")
        }),
        TargetProperty.fromDict({
            "name": "P29275",
            "task": "REGRESSION",
            "imputer": SimpleImputer(strategy="median")
        }),
        TargetProperty.fromDict({
            "name": "P30542",
            "task": "REGRESSION",
            "imputer": SimpleImputer(strategy="median")
        })
    ],
]

# assessors
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
