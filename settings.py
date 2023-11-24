from qsprpred.data.data import MoleculeTable, TargetProperty
from qsprpred.data.descriptors.sets import FingerprintSet
from qsprpred.data.sampling.splits import RandomSplit, ClusterSplit, ScaffoldSplit
from qsprpred.data.sources.papyrus import Papyrus
from qsprpred.models.assessment_methods import CrossValAssessor, TestSetAssessor
from qsprpred.models.sklearn import SklearnModel
from qsprpred.models.tasks import TargetTasks
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from benchmark_settings import DataPrepSettings, BenchmarkSettings

NAME = "ExampleBenchmark"  # name of the benchmark
N_REPLICAS = 30  # number of repetitions per experiment
SEED = 42  # seed for random operations (seed for all random states)
DATA_DIR = f"./data/{NAME}"  # directory to store data
N_PROC = 12  # number of processes to use for parallelization


class PapyrusForBenchmark(Papyrus):

    def __init__(self, acc_keys: list[str]):
        super().__init__(
            data_dir=f"{DATA_DIR}/sets",
            version="05.6",
            plus_only=True
        )
        self.acc_keys = acc_keys

    def getData(
        self,
        name: str,
        **kwargs,
    ) -> MoleculeTable:
        return super().getData(
            name=name,
            acc_keys=self.acc_keys,
            quality="high",
            activity_types="all",
            drop_duplicates=True,
            use_existing=True,
            **kwargs,
        )


# data sources
DATA_SOURCES = [
    PapyrusForBenchmark(["P30542"]),  # A1
    PapyrusForBenchmark(["P29274"]),  # A2A
    PapyrusForBenchmark(["P29275"]),  # A2B
    PapyrusForBenchmark(["P0DMS8"]),  # A3
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
        descriptors=DESCRIPTORS,
    ),
    DataPrepSettings(
        split=ClusterSplit(test_fraction=0.2),
        descriptors=DESCRIPTORS,
    ),
    DataPrepSettings(
        split=ScaffoldSplit(test_fraction=0.2),
        descriptors=DESCRIPTORS,
    ),
]

# models
MODELS = [
    SklearnModel(alg=ExtraTreesClassifier, base_dir=f"{DATA_DIR}/models"),
    SklearnModel(alg=XGBClassifier, base_dir=f"{DATA_DIR}/models"),
    SklearnModel(alg=SVC, base_dir=f"{DATA_DIR}/models"),
    SklearnModel(alg=MLPClassifier, base_dir=f"{DATA_DIR}/models"),
    SklearnModel(alg=GaussianNB, base_dir=f"{DATA_DIR}/models"),
]

# assessors
ASSESSORS = [
    CrossValAssessor(scoring="roc_auc", split=prep.split) for prep in DATA_PREPS
] + [
    CrossValAssessor(scoring="matthews_corrcoef", split=prep.split) for prep in DATA_PREPS
] + [
    CrossValAssessor(scoring="recall", split=prep.split) for prep in DATA_PREPS
] + [
    CrossValAssessor(scoring="precision", split=prep.split) for prep in DATA_PREPS
] + [
    TestSetAssessor(scoring="roc_auc"),
    TestSetAssessor(scoring="matthews_corrcoef"),
    TestSetAssessor(scoring="recall"),
    TestSetAssessor(scoring="precision"),
]

# benchmark settings
SETTINGS = BenchmarkSettings(
    name=NAME,
    n_replicas=N_REPLICAS,
    random_seed=SEED,
    data_sources=DATA_SOURCES,
    target_props=TARGET_PROPS,
    prep_settings=DATA_PREPS,
    models=MODELS,
    assessors=ASSESSORS,
    optimizers=[],  # no hyperparameter optimization
)
SETTINGS.toFile(f"{DATA_DIR}/{NAME}.json")


