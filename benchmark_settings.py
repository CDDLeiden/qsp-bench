import json
import logging

import itertools
import random
from typing import Callable, Generator

import numpy as np
import pandas as pd

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
        name (str): name of the benchmark
        data_source (DataSource): data source to use
        target_props (TargetProperty | dict): target property to use
        prep_settings (DataPrepSettings): data preparation settings to use
        model (QSPRModel): model to use
        assessors (ModelAssessor): model assessor to use
        optimizer (HyperparameterOptimization): hyperparameter optimizer to use
        random_seed (int): seed for random operations
        ds (QSPRDataset): data set used for this replica
    """

    _notJSON = ["model"]

    idx: int
    name: str
    data_source: DataSource
    descriptors: list[DescriptorSet]
    target_props: list[TargetProperty]
    prep_settings: DataPrepSettings
    model: QSPRModel
    optimizer: HyperparameterOptimization
    assessors: list[ModelAssessor]
    random_seed: int
    ds: QSPRDataset | None = None

    def __getstate__(self):
        o_dict = super().__getstate__()
        o_dict["model"] = self.model.save()
        o_dict["ds"] = None
        return o_dict

    def __setstate__(self, state):
        super().__setstate__(state)
        self.model = QSPRModel.fromFile(state["model"])
        self.ds = None

    @property
    def id(self):
        """Returns the identifier of this replica."""
        return f"{self.name}_{self.random_seed}"

    def create_dataset(self, reload=False):
        # get the basics set from data source
        self.ds = self.data_source.getDataSet(
            self.target_props,
            overwrite=reload,
            random_state=self.random_seed
        )
        # add descriptors
        self.add_descriptors(reload=reload)

    def add_descriptors(
            self,
            reload: bool = False
    ):
        # generate name for the data with descriptors
        desc_id = "_".join([str(d) for d in self.descriptors])
        # tp_id = "_".join([tp.name for tp in ds.targetProperties])
        ds_desc_name = f"{self.ds.name}_{desc_id}"
        # create or reload the data set
        try:
            ds_descs = QSPRDataset(
                name=ds_desc_name,
                store_dir=self.ds.baseDir,
                target_props=self.target_props,
                random_state=self.random_seed
            )
        except ValueError:
            logging.warning(f"Data set {ds_desc_name} not found. It will be created.")
            ds_descs = QSPRDataset(
                name=ds_desc_name,
                store_dir=self.ds.baseDir,
                target_props=self.target_props,
                random_state=self.random_seed,
                df=self.ds.getDF(),
            )
        # calculate descriptors if necessary
        if not ds_descs.hasDescriptors or reload:
            logging.info(f"Calculating descriptors for {ds_descs.name}.")
            desc_calculator = MoleculeDescriptorsCalculator(
                desc_sets=self.descriptors
            )
            ds_descs.addDescriptors(desc_calculator, recalculate=True)
            ds_descs.save()
        self.ds = ds_descs
        self.ds.save()

    def prep_dataset(self):
        self.ds.prepareDataset(
            **self.prep_settings.__dict__,
        )

    def create_report(self):
        self.initModel()
        results = self.run_assessment()
        results["ModelFile"] = self.model.save()
        results["Algorithm"] = self.model.alg.__name__
        results["AlgorithmParams"] = json.dumps(self.model.parameters)
        results["ReplicaID"] = self.id
        results["DataSet"] = self.ds.name
        out_file = f"{self.model.outPrefix}_replica.json"
        for assessor in self.assessors:
            # FIXME: some problems in monitor serialization now prevent this
            assessor.monitor = None
        self.model.data = None  # FIXME: model now does not support data serialization
        results["ReplicaFile"] = self.toFile(out_file)
        return results

    def initModel(self):
        self.model.name = self.id
        self.model.initFromData(self.ds)
        self.model.initRandomState(self.random_seed)
        if self.optimizer is not None:
            self.optimizer.optimize(self.model)

    def run_assessment(self):
        results = None
        for assessor in self.assessors:
            scores = assessor(self.model, save=True)
            scores = pd.DataFrame({
                "Assessor": assessor.__class__.__name__,
                "ScoreFunc": assessor.scoreFunc.name,
                "Score": scores,
            })
            if results is None:
                results = scores
            else:
                results = pd.concat([results, scores])
        return results


@dataclass
class BenchmarkSettings(JSONSerializable):
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

    @property
    def n_runs(self):
        """Returns the total number of benchmarking runs."""
        self.checkConsistency()
        ret = (self.n_replicas * len(self.data_sources)
                * len(self.descriptors) * len(self.target_props)
                * len(self.prep_settings) * len(self.models)
               )
        if len(self.optimizers) > 0:
            ret *= len(self.optimizers)
        return ret

    def get_seed_list(self, seed: int) -> list[int]:
        """
        Get a list of seeds for the replicas.

        Args:
            seed(int): master seed to generate the list of seeds from

        Returns:
            list[int]: list of seeds for the replicas

        """
        random.seed(seed)
        return random.sample(range(2**32 - 1), self.n_runs)

    def checkConsistency(self):
        assert len(self.data_sources) > 0, "No data sources defined."
        assert len(self.descriptors) > 0, "No descriptors defined."
        assert len(self.target_props) > 0, "No target properties defined."
        assert len(self.prep_settings) > 0, "No data preparation settings defined."
        assert len(self.models) > 0, "No models defined."
        assert len(self.assessors) > 0, "No model assessors defined."

    def iter_replicas(self) -> Generator[Replica, None, None]:
        np.random.seed(self.random_seed)
        # generate all combinations of settings with itertools
        self.checkConsistency()
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
        seeds = self.get_seed_list(self.random_seed)
        for idx, settings in enumerate(product):
            yield Replica(
                *settings,
                random_seed=seeds[idx],
                assessors=self.assessors
            )
