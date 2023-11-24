import hashlib
import itertools
from typing import Callable

import numpy as np
from qsprpred.data.data import TargetProperty
from qsprpred.data.descriptors.sets import DescriptorSet
from qsprpred.data.processing.feature_standardizers import SKLearnStandardizer
from qsprpred.data.sampling.splits import DataSplit
from qsprpred.data.sources.data_source import DataSource
from qsprpred.models.assessment_methods import ModelAssessor
from qsprpred.models.hyperparam_optimization import HyperparameterOptimization
from qsprpred.models.models import QSPRModel
from qsprpred.utils.serialization import JSONSerializable
from dataclasses import dataclass


@dataclass
class DataPrepSettings:
    descriptors: list[list[DescriptorSet]]
    split: DataSplit = None
    smiles_standardizer: str | Callable = "chembl"
    feature_filters: list = None
    feature_standardizer: SKLearnStandardizer = None
    feature_fill_value: float = 0.0


@dataclass
class ReplicaSettings(JSONSerializable):
    """Class that determines settings for a single replica of a benchmarking run.

    Attributes:
        idx (int): number of the replica
        benchmark_name (str): name of the benchmark
        data_source (DataSource): data source to use
        target_props (TargetProperty | dict): target property to use
        prep_settings (DataPrepSettings): data preparation settings to use
        model (QSPRModel): model to use
        assessor (ModelAssessor): model assessor to use
        optimizer (HyperparameterOptimization): hyperparameter optimizer to use
    """

    _notJSON = ["model"]

    idx: int
    benchmark_name: str
    random_seed: int
    data_source: DataSource
    target_props: list[TargetProperty]
    prep_settings: DataPrepSettings
    model: QSPRModel
    assessor: ModelAssessor
    optimizer: HyperparameterOptimization

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["model"] = self.model.save()
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.model = QSPRModel.fromFile(state["model"])

    @property
    def id(self):
        """Returns the id of the replica, which is an md5 hash of its JSON form."""
        return hashlib.md5(self.toJSON())


@dataclass
class BenchmarkSettings(JSONSerializable):
    """Class that determines settings for a benchmarking run.

    Attributes:
        name (str): name of the benchmark
        n_replicas (int): number of repetitions per experiment
        random_seed (int): seed for random operations
        data_sources (list[DataSource]): data sources to use
        target_props (list[list[TargetProperty]]): target properties to use
        prep_settings (list[DataPrepSettings]): data preparation settings to use
        models (list[QSPRModel]): models to use
        assessors (list[ModelAssessor]): model assessors to use
        optimizers (list[HyperparameterOptimization]): hyperparameter optimizers to use
    """

    _notJSON = ["models"]

    name: str
    n_replicas: int
    random_seed: int
    data_sources: list[DataSource]
    target_props: list[list[TargetProperty]]
    prep_settings: list[DataPrepSettings]
    models: list[QSPRModel]
    assessors: list[ModelAssessor]
    optimizers: list[HyperparameterOptimization]

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["models"] = [model.save() for model in self.models]
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.models = [QSPRModel.fromFile(model) for model in state["models"]]

    def iter_replicas(self):
        np.random.seed(self.random_seed)
        # generate all combinations of settings with itertools
        product = itertools.product(
            [x+1 for x in range(self.n_replicas)],
            [self.name],
            [np.random.randint(1, 2**32-1) for _ in range(self.n_replicas)],
            self.data_sources,
            self.target_props,
            self.prep_settings,
            self.models,
            self.assessors,
            self.optimizers,
        )
        for settings in product:
            yield ReplicaSettings(
                *settings
            )


