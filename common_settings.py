import os
import shutil

from qsprpred.data.data import TargetProperty
from qsprpred.data.descriptors.sets import FingerprintSet
from qsprpred.data.sampling.splits import RandomSplit
from qsprpred.models.assessment_methods import TestSetAssessor
from qsprpred.models.sklearn import SklearnModel
from qsprpred.models.tasks import TargetTasks
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from qsprpred.benchmarks import DataPrepSettings, BenchmarkSettings
from tools import PapyrusForBenchmark

START_FRESH = False  # set True to run all replicas from scratch
RESET_MODELS = False  # set True to reset all models
NAME = "ExampleBenchmark"  # name of the benchmark
N_REPLICAS = 3  # number of repetitions per experiment
SEED = 42  # randomSeed for random operations (randomSeed for all random states)
DATA_DIR = f"./data/{NAME}"  # directory to store data
MODELS_DIR = f"{DATA_DIR}/models"  # directory to store models
N_PROC = 12  # number of processes to use for parallelization
RESULTS_FILE = f"{DATA_DIR}/results.tsv"  # file to store results

# data sources
DATA_SOURCES = [
    PapyrusForBenchmark(["P30542"], f"{DATA_DIR}/sets"),  # A1
    PapyrusForBenchmark(["P29274"], f"{DATA_DIR}/sets"),  # A2A
    # PapyrusForBenchmark(["P29275"], f"{DATA_DIR}/sets"),  # A2B
    # PapyrusForBenchmark(["P0DMS8"], f"{DATA_DIR}/sets"),  # A3
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
    # DataPrepSettings(
    #     split=ClusterSplit(test_fraction=0.2),
    # ),
    # DataPrepSettings(
    #     split=ScaffoldSplit(test_fraction=0.2),
    # ),
]

# models
MODELS = [
    SklearnModel(
        name="ExtraTreesClassifier",
        alg=ExtraTreesClassifier,
        base_dir=MODELS_DIR,
    ),
    SklearnModel(
        name="GaussianNB",
        alg=GaussianNB,
        base_dir=MODELS_DIR,
    ),
    SklearnModel(
        name="MLPClassifier",
        alg=MLPClassifier,
        base_dir=f"{DATA_DIR}/models",
    ),
    SklearnModel(
        name="SVC",
        alg=SVC,
        base_dir=f"{DATA_DIR}/models",
        parameters={
            "probability": True,
            "max_iter": 1000,
        }
    ),
    SklearnModel(
        name="XGBClassifier",
        alg=XGBClassifier,
        base_dir=f"{DATA_DIR}/models",
    ),
]

# assessors
ASSESSORS = [
    TestSetAssessor(scoring="roc_auc"),
    TestSetAssessor(scoring="matthews_corrcoef", use_proba=False),
    # TestSetAssessor(scoring="recall", use_proba=False),
    # TestSetAssessor(scoring="precision", use_proba=False),
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
    optimizers=[],  # no hyperparameter optimization
)

# create data directory
if os.path.exists(DATA_DIR) and START_FRESH:
    shutil.rmtree(DATA_DIR)
if os.path.exists(MODELS_DIR) and RESET_MODELS:
    shutil.rmtree(MODELS_DIR)
if os.path.exists(RESULTS_FILE) and RESET_MODELS:
    os.remove(RESULTS_FILE)
