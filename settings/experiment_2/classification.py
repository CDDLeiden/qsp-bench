from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd

import deepchem as dc
from xgboost import XGBClassifier

from qsprpred import TargetProperty, TargetTasks
from qsprpred.benchmarks import BenchmarkSettings
from qsprpred.data import MoleculeTable
from qsprpred.data.sources import DataSource
from qsprpred.extra.gpu.models.dnn import DNNModel
from qsprpred.extra.gpu.models.neural_network import STFullyConnected
from qsprpred.extra.models.random import RandomModel, RatioDistributionAlgorithm
from qsprpred.models import SklearnModel, TestSetAssessor
from settings.base import *

sets_to_loads = {
    "HIV": dc.molnet.load_hiv,
    "BACE": dc.molnet.load_bace_classification,
    "BBBP": dc.molnet.load_bbbp,
}
if not os.environ.get("QSPBENCH_MOLNETSET"):
    raise ValueError(
        f"Please set the QSPBENCH_MOLNETSET environment variable. "
        f"Options are: {list(sets_to_loads.keys())}"
    )
DATASET = os.environ["QSPBENCH_MOLNETSET"]
prop_name = sets_to_loads[DATASET](data_dir=DATA_DIR, save_dir=DATA_DIR)[0][0]


class MoleculeNetSource(DataSource):
    def __init__(
        self,
        name: str,
        store_dir: str,
        smiles_col: str = "smiles",
        n_jobs: int = os.cpu_count(),
        n_samples: int | None = None,
    ):
        self.storeDir = store_dir
        self.name = name
        self.smilesCol = smiles_col
        self.nJobs = n_jobs
        self.nSamples = n_samples

    def getData(self, name: str | None = None, **kwargs) -> MoleculeTable:
        name = name or self.name
        dataset_path = os.path.join(self.storeDir, f"{self.name}.csv")
        df = pd.read_csv(dataset_path, header=0)
        if self.nSamples is not None:
            df = df.sample(self.nSamples)
        return MoleculeTable(
            df=df,
            smiles_col=self.smilesCol,
            name=name,
            store_dir=self.storeDir,
            n_jobs=self.nJobs,
            **kwargs,
        )


# data sources
DATA_SOURCES = [
    MoleculeNetSource(DATASET, DATA_DIR, n_jobs=N_PROC, n_samples=N_SAMPLES),
]

# target properties
TARGET_PROPS = [
    [TargetProperty(prop_name, TargetTasks.SINGLECLASS, th="precomputed")],
]

MODELS = [
    SklearnModel(name="XGBClassifier", alg=XGBClassifier, base_dir=MODELS_DIR),
    SklearnModel(name="GaussianNB", alg=GaussianNB, base_dir=MODELS_DIR),
    SklearnModel(
        name="SVC", alg=SVC, base_dir=MODELS_DIR, parameters={"probability": True}
    ),
    SklearnModel(name="MLPClassifier", alg=MLPClassifier, base_dir=MODELS_DIR),
    RandomModel(
        name=f"{NAME}_MedianModel",
        base_dir=MODELS_DIR,
        alg=RatioDistributionAlgorithm,
    ),
    # DNNModel(
    #     name="DNNModel",
    #     alg=STFullyConnected,
    #     base_dir=MODELS_DIR,
    # ),
]

# assessors
ASSESSORS = [
    TestSetAssessor(scoring="roc_auc"),
    TestSetAssessor(scoring="matthews_corrcoef", use_proba=False),
    TestSetAssessor(scoring="recall", use_proba=False),
    TestSetAssessor(scoring="precision", use_proba=False),
    TestSetAssessor(scoring="f1", use_proba=False),
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
