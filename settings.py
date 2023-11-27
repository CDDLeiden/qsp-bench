import os

from qsprpred.data.data import TargetProperty
from qsprpred.data.descriptors.sets import FingerprintSet
from qsprpred.data.sampling.splits import RandomSplit, ClusterSplit, ScaffoldSplit
from qsprpred.models.assessment_methods import CrossValAssessor, TestSetAssessor
from qsprpred.models.sklearn import SklearnModel
from qsprpred.models.tasks import TargetTasks
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from benchmark_settings import DataPrepSettings, Benchmark
from tools import PapyrusForBenchmark

NAME = "ExampleBenchmark"  # name of the benchmark
N_REPLICAS = 30  # number of repetitions per experiment
SEED = 42  # seed for random operations (seed for all random states)
DATA_DIR = f"./data/{NAME}"  # directory to store data
N_PROC = 1  # number of processes to use for parallelization

# data sources
DATA_SOURCES = [
    PapyrusForBenchmark(["P30542"], f"{DATA_DIR}/sets"),  # A1
    PapyrusForBenchmark(["P29274"], f"{DATA_DIR}/sets"),  # A2A
    PapyrusForBenchmark(["P29275"], f"{DATA_DIR}/sets"),  # A2B
    PapyrusForBenchmark(["P0DMS8"], f"{DATA_DIR}/sets"),  # A3
]

# target properties
TARGET_PROPS = [
    [TargetProperty("pchembl_value_Median", TargetTasks.SINGLECLASS, th=[7])],
]

# descriptors
DESCRIPTORS = [
    [FingerprintSet("MorganFP", radius=3, nBits=2048)],
]

# data preparation settings
DATA_PREPS = [
    DataPrepSettings(
        split=RandomSplit(test_fraction=0.2),
    ),
    DataPrepSettings(
        split=ClusterSplit(test_fraction=0.2),
    ),
    DataPrepSettings(
        split=ScaffoldSplit(test_fraction=0.2),
    ),
]

# models
MODELS = [
    SklearnModel(
        name="ExtraTreesClassifier",
        alg=ExtraTreesClassifier,
        base_dir=f"{DATA_DIR}/models"
    ),
    # SklearnModel(
    #     name="GaussianNB",
    #     alg=GaussianNB,
    #     base_dir=f"{SETTINGS_DIR}/models"
    # ),
    # SklearnModel(
    #     name="MLPClassifier",
    #     alg=MLPClassifier,
    #     base_dir=f"{SETTINGS_DIR}/models"
    # ),
    # SklearnModel(
    #     name="SVC",
    #     alg=SVC,
    #     base_dir=f"{SETTINGS_DIR}/models"
    # ),
    # SklearnModel(
    #     name="XGBClassifier",
    #     alg=XGBClassifier,
    #     base_dir=f"{SETTINGS_DIR}/models"
    # ),
]

# assessors
ASSESSORS = [
    TestSetAssessor(scoring="roc_auc"),
    # TestSetAssessor(scoring="matthews_corrcoef"),
    # TestSetAssessor(scoring="recall"),
    # TestSetAssessor(scoring="precision"),
]

# benchmark settings
SETTINGS = Benchmark(
    name=NAME,
    n_replicas=N_REPLICAS,
    random_seed=SEED,
    data_sources=DATA_SOURCES,
    descriptors=DESCRIPTORS,
    target_props=TARGET_PROPS,
    prep_settings=DATA_PREPS,
    models=MODELS,
    assessors=ASSESSORS,
    optimizers=[],  # no hyperparameter optimization
)
SETTINGS.toFile(f"{DATA_DIR}/{NAME}.json")


