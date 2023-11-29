import logging

import itertools
import random
from typing import Callable, Generator

import numpy as np
from qsprpred.data.data import TargetProperty, QSPRDataset
from qsprpred.data.descriptors.calculators import MoleculeDescriptorsCalculator
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
    split: DataSplit = None
    smiles_standardizer: str | Callable = "chembl"
    feature_filters: list = None
    feature_standardizer: SKLearnStandardizer = None
    feature_fill_value: float = 0.0


@dataclass
class Replica(JSONSerializable):
    """Class that determines settings for a single replica of a benchmarking run.

    Attributes:
        idx (int): number of the replica
        benchmark_name (str): name of the benchmark
        data_source (DataSource): data source to use
        target_props (TargetProperty | dict): target property to use
        prep_settings (DataPrepSettings): data preparation settings to use
        model (QSPRModel): model to use
        assessors (ModelAssessor): model assessor to use
        optimizer (HyperparameterOptimization): hyperparameter optimizer to use
        is_finished (bool): whether the replica has finished benchmarking
    """

    _notJSON = ["model"]

    idx: int
    benchmark_name: str
    data_source: DataSource
    descriptors: list[DescriptorSet]
    target_props: list[TargetProperty]
    prep_settings: DataPrepSettings
    model: QSPRModel
    optimizer: HyperparameterOptimization
    assessors: list[ModelAssessor]
    random_seed: int
    is_finished: bool = False

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["model"] = self.model.save()
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.model = QSPRModel.fromFile(state["model"])

    @property
    def id(self):
        """Returns the identifier of this replica."""
        return f"{self.benchmark_name}_{self.random_seed}"

    def get_dataset(self, reload=False):
        # get the basics set from data source
        ds = self.data_source.getDataSet(
            self.target_props,
            overwrite=reload,
            random_state=self.random_seed
        )
        if reload:
            ds.save()
        # add descriptors
        ds = self.add_descriptors(ds, reload=reload)
        return ds

    def add_descriptors(
            self,
            ds: QSPRDataset,
            reload: bool = False
    ) -> QSPRDataset:
        # generate name for the data with descriptors
        desc_id = "_".join([str(d) for d in self.descriptors])
        # tp_id = "_".join([tp.name for tp in ds.targetProperties])
        ds_desc_name = f"{ds.name}_{desc_id}"
        # create or reload the data set
        try:
            ds_prepped = QSPRDataset(
                name=ds_desc_name,
                store_dir=ds.baseDir,
                target_props=self.target_props,
                random_state=self.random_seed
            )
        except ValueError:
            logging.warning(f"Data set {ds_desc_name} not found. It will be created.")
            ds_prepped = QSPRDataset(
                name=ds_desc_name,
                store_dir=ds.baseDir,
                target_props=self.target_props,
                random_state=self.random_seed,
                df=ds.getDF(),
            )
            ds_prepped.save()
        # calculate descriptors if necessary
        if not ds_prepped.hasDescriptors or reload:
            desc_calculator = MoleculeDescriptorsCalculator(
                desc_sets=self.descriptors
            )
            ds_prepped.addDescriptors(desc_calculator, recalculate=True)
            ds_prepped.save()
        return ds_prepped

    def prep_dataset(self, ds: QSPRDataset) -> QSPRDataset:
        ds.prepareDataset(
            **self.prep_settings.__dict__,
        )
        return ds


@dataclass
class Benchmark(JSONSerializable):
    """Class that determines settings for a benchmarking run.

    Attributes:
        name (str): name of the benchmark
        n_replicas (int): number of repetitions per experiment
        random_seed (int): seed for random operations
        data_sources (list[DataSource]): data sources to use
        descriptors (list[list[DescriptorSet]]): descriptors to use
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
    descriptors: list[list[DescriptorSet]]
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

    @staticmethod
    def get_pseudo_random_list(seed: int, n: int) -> list:
        """Returns a list of pseudo-random numbers.

        Args:
            seed (int): seed for random operations
            n (int): number of random numbers to generate

        Returns:
            list: list of pseudo-random numbers
        """
        random.seed(seed)
        return random.sample(range(2**32 - 1), n)

    def iter_replicas(self) -> Generator[Replica, None, None]:
        np.random.seed(self.random_seed)
        # generate all combinations of settings with itertools
        assert len(self.data_sources) > 0, "No data sources defined."
        assert len(self.descriptors) > 0, "No descriptors defined."
        assert len(self.target_props) > 0, "No target properties defined."
        assert len(self.prep_settings) > 0, "No data preparation settings defined."
        assert len(self.models) > 0, "No models defined."
        assert len(self.assessors) > 0, "No model assessors defined."
        indices = [x+1 for x in range(self.n_replicas)]
        optimizers = self.optimizers if len(self.optimizers) > 0 else [None]
        product = itertools.product(
            indices,
            [self.name],
            self.data_sources,
            self.descriptors,
            self.target_props,
            self.prep_settings,
            self.models,
            optimizers,
        )
        n_items = (len(indices) * len(self.data_sources)
                   * len(self.descriptors) * len(self.target_props)
                   * len(self.prep_settings) * len(self.models)
                   * len(optimizers))
        seeds = self.get_pseudo_random_list(self.random_seed, n_items)
        for idx, settings in enumerate(product):
            yield Replica(
                *settings,
                random_seed=seeds[idx],
                assessors=self.assessors
            )


