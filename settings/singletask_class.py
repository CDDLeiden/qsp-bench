from .base import *
from .data_sources import PapyrusForBenchmark
from qsprpred import TargetProperty, TargetTasks
from qsprpred.models import SklearnModel, TestSetAssessor
from qsprpred.benchmarks import BenchmarkSettings
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier

# data sources
DATA_SOURCES = [
    PapyrusForBenchmark(
        ["P30542", "P29274", "P29275", "P0DMS8"],
        f"{DATA_DIR}/sets", n_samples=100
    ),
]

MODELS=[
    SklearnModel(
        name="GaussianNB",
        alg=GaussianNB,
        base_dir=f"{MODELS_DIR}/models",
    ),
    SklearnModel(
        name="ExtraTreesClassifier",
        alg=ExtraTreesClassifier,
        base_dir=f"{MODELS_DIR}/models",
    )
]

TARGET_PROPS=[
    # one or more properties to model
    [
        TargetProperty.fromDict(
            {
                "name": "pchembl_value_Mean",
                "task": TargetTasks.SINGLECLASS,
                "th": [6.5]
            }
        )
    ],
]

ASSESSORS=[
    TestSetAssessor(scoring="roc_auc"),
    TestSetAssessor(scoring="matthews_corrcoef", use_proba=False),
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
