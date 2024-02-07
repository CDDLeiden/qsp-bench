import logging

import os
from importlib import import_module

from qsprpred.logs import logger, setLogger
from qsprpred.benchmarks import BenchmarkSettings, BenchmarkRunner


if "QSPBENCH_SETTINGS" not in os.environ:
    raise ValueError(
        "Please set the QSPBENCH_SETTINGS environment "
        "variable with an import path to your settings module."
    )
settings_module = os.environ["QSPBENCH_SETTINGS"]
os.environ["QSPBENCH_SETTINGS"] = settings_module
settings_module = import_module(settings_module)
settings = BenchmarkSettings(
    name=settings_module.NAME,
    n_replicas=settings_module.N_REPLICAS,
    random_seed=settings_module.SEED,
    data_sources=settings_module.DATA_SOURCES,
    descriptors=settings_module.DESCRIPTORS,
    target_props=settings_module.TARGET_PROPS,
    prep_settings=settings_module.DATA_PREPS,
    models=settings_module.MODELS,
    assessors=settings_module.ASSESSORS,
)
runner = BenchmarkRunner(
    settings,
    settings_module.N_PROC,
    settings_module.DATA_DIR,
    settings_module.RESULTS_FILE,
)
runner.run(raise_errors=True)
