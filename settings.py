import os
import pandas as pd
import shutil

import deepchem as dc

from qsprpred import TargetProperty, TargetTasks
from qsprpred.benchmarks import BenchmarkSettings, DataPrepSettings
from qsprpred.data import ClusterSplit, MoleculeTable, QSPRDataset, RandomSplit
from qsprpred.data.descriptors.sets import FingerprintSet
from qsprpred.data.processing.feature_filters import LowVarianceFilter
from qsprpred.data.sources import DataSource
from qsprpred.data.descriptors.calculators import MoleculeDescriptorsCalculator
from qsprpred.extra.gpu.models.dnn import DNNModel
from qsprpred.extra.gpu.models.neural_network import STFullyConnected
from qsprpred.models import SklearnModel, TestSetAssessor
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier, XGBRegressor



BASE_DIR = "/home/remco/projects/qsprpred_bench/e2_reg_qsp-bench"  # directory to store all benchmarking results and files
os.makedirs(BASE_DIR, exist_ok=True)

# load dataset
dataset_delaney = dc.molnet.load_lipo(data_dir=BASE_DIR)
dataset_path = BASE_DIR + '/Lipophilicity.csv'
dataset_smiles_col = 'smiles'
dataset_property_col = "exp"


class MoleculeNetSource(DataSource):
    def __init__(self, name: str, store_dir: str):
        self.name = name
        self.storeDir = store_dir

    def getData(self, name: str | None = None, **kwargs) -> MoleculeTable:
        name = name or self.name
        return MoleculeTable(
            df=pd.read_csv(dataset_path),
            smiles_col=dataset_smiles_col,
            name=name,
            store_dir=self.storeDir,
            **kwargs
        )

# benchmark settings    
NAME = "e2_reg_benchmark"    
START_FRESH = True  # set True to run all replicas from scratch
RESET_MODELS = True  # set True to reset all models
N_REPLICAS = 5  # number of repetitions per experiment
SEED = 42  # randomSeed for random operations (randomSeed for all random states)
DATA_DIR = f"{BASE_DIR}/data/"  # directory to store data
MODELS_DIR = f"{DATA_DIR}/models"  # directory to store models
N_PROC = 12  # number of processes to use for parallelization
RESULTS_FILE = f"{DATA_DIR}/results.tsv"  # file to store results



# data sources
DATA_SOURCES = [
    MoleculeNetSource(NAME, DATA_DIR)
]

# target properties
TARGET_PROPS = [
    [TargetProperty(dataset_property_col, TargetTasks.REGRESSION)],
]

# descriptors
DESCRIPTORS = [
    [FingerprintSet("MorganFP", radius=3, nBits=2048)],
]

# data preparation settings
DATA_PREPS = [
    DataPrepSettings(
        split=RandomSplit(test_fraction=0.2),
        feature_filters=[LowVarianceFilter(0.05)],
        feature_standardizer=StandardScaler()
    ),
    DataPrepSettings(
        split=ClusterSplit(test_fraction=0.2),
        feature_filters=[LowVarianceFilter(0.05)],
        feature_standardizer=StandardScaler()
    )
]

# DNNModel requires a dataset 
dataset = QSPRDataset.fromTableFile(
            filename=dataset_path,
            sep=',' ,
            store_dir=DATA_DIR,
            name=f'{BASE_DIR}/{NAME}',
            smiles_col=dataset_smiles_col,
            target_props=[{"name": dataset_property_col, "task": "REGRESSION"}],
            random_state=SEED
        )

dataset.prepareDataset(
            split=RandomSplit(test_fraction=0.2),
            feature_calculators=[MoleculeDescriptorsCalculator(desc_sets = [FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=2048)])],
            feature_standardizer=StandardScaler(),
            feature_filters=[LowVarianceFilter(0.05)])

MODELS = [
    SklearnModel(
        name="XGBRegressor",
        alg=XGBRegressor,
        base_dir=MODELS_DIR
    ),
    #SklearnModel(
    #    name="XGBClassifier",
    #    alg=XGBClassifier,
    #    base_dir=MODELS_DIR
    #),
#    DNNModel(
#        name="DNNModel",
#        alg=STFullyConnected,
#        data=dataset,
        #patience=3,
        #tol=0.01,
#        base_dir=MODELS_DIR,
    #   #device=torch.device("cpu")
#    )
]

# assessors
ASSESSORS = [
    TestSetAssessor('r2'),
    # TestSetAssessor(scoring="roc_auc"),
    # TestSetAssessor(scoring="matthews_corrcoef", use_proba=False),
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


