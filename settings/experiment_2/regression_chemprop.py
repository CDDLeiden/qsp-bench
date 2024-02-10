from qsprpred.data.descriptors.sets import SmilesDesc
from qsprpred.extra.gpu.models.chemprop import ChempropModel
from settings.experiment_2.regression import *

DESCRIPTORS = [
    [SmilesDesc()],
]

MODELS = [
    ChempropModel(
        name=f"{NAME}_Chemprop",
        base_dir=MODELS_DIR,
    ),
]

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
