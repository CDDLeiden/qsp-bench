from common_settings import DATA_DIR, MODELS_DIR
from tools import PapyrusForBenchmark
from qsprpred.benchmarks import DataPrepSettings
from qsprpred.data.sampling.splits import RandomSplit, ClusterSplit
from qsprpred.extra.gpu.models.pyboost import PyBoostModel
from qsprpred.models.sklearn import SklearnModel
from sklearn.naive_bayes import GaussianNB

def _multitask_common_settings():
    # data sources
    DATA_SOURCES = [
        PapyrusForBenchmark(["P30542"], f"{DATA_DIR}/sets"),  # A1
        PapyrusForBenchmark(["P29274"], f"{DATA_DIR}/sets"),  # A2A
        # PapyrusForBenchmark(["P29275"], f"{DATA_DIR}/sets"),  # A2B
        # PapyrusForBenchmark(["P0DMS8"], f"{DATA_DIR}/sets"),  # A3
    ]
        # models
    MODELS = [
        SklearnModel(
            name="GaussianNB",
            alg=GaussianNB,
            base_dir=MODELS_DIR,
        ),
        PyBoostModel(
            name="PyBoost",
            base_dir=MODELS_DIR,
        )
    ]
    return {
        "data_sources": DATA_SOURCES,
        "models": MODELS,
    }

def randomsplit_settings():
    DATA_PREPS = [
        DataPrepSettings(
            split=RandomSplit(test_fraction=0.2),
        ),
    ]

    return {
        **_multitask_common_settings(),
        "prep_settings": DATA_PREPS
    }

def clustersplit_settings():
    DATA_PREPS = [
       DataPrepSettings(
            split=ClusterSplit(test_fraction=0.2),
        ),
    ]

    return {
        **_multitask_common_settings(),
        "prep_settings": DATA_PREPS
    }
