import json

import logging
import os
import random
import string
import subprocess
import sys
import warnings
from copy import deepcopy
from itertools import product

import pandas as pd
from matplotlib import pyplot as plt
from qsprpred.data.data import QSPRDataset, MoleculeTable
from qsprpred.data.descriptors.calculators import MoleculeDescriptorsCalculator
from qsprpred.data.descriptors.sets import FingerprintSet
from qsprpred.data.sources.papyrus import Papyrus
from qsprpred.models.sklearn import SklearnModel
from qsprpred.plotting.classification import MetricsPlot

from benchmark_settings import Replica


class PapyrusForBenchmark(Papyrus):

    def __init__(self, acc_keys: list[str], data_dir: str):
        super().__init__(
            data_dir=data_dir,
            version="05.6",
            plus_only=True
        )
        self.acc_keys = sorted(acc_keys)

    def getData(
        self,
        name: str | None = None,
        **kwargs,
    ) -> MoleculeTable:
        return super().getData(
            name=name or "_".join(self.acc_keys),
            acc_keys=self.acc_keys,
            quality="high",
            activity_types="all",
            drop_duplicates=True,
            use_existing=True,
            **kwargs,
        )


def benchmark_replica(ds: QSPRDataset, replica: Replica):
    replica = deepcopy(replica)
    model = replica.model
    model.name = replica.id
    model.initFromData(ds)
    model.initRandomState(replica.random_seed)
    if replica.optimizer is not None:
        replica.optimizer.optimize(model)
    results = None
    for assessor in replica.assessors:
        scores = assessor(model, save=True)
        scores = pd.DataFrame({
            "Assessor": assessor.__class__.__name__,
            "ScoreFunc": assessor.scoreFunc.name,
            "Score": scores,
        })
        if results is None:
            results = scores
        else:
            results = pd.concat([results, scores])
    replica.is_finished = True
    return results, replica


def get_random_string():
    return ''.join(random.choice(string.ascii_letters) for i in range(10))


def save_temp(ds, index):
    df = ds.getDF()
    os.makedirs("temp", exist_ok=True)
    name = get_random_string()
    name = f"temp/{name}_temp.smi"
    df.loc[index, [ds.smilesCol] + ds.indexCols].to_csv(name, index=False, header=False, sep=" ")
    return name


def get_AVE_bias(ds, target_prop):
    df = ds.getDF()
    actives_train = ds.X[df.loc[ds.X.index, target_prop] == True].index
    inactives_train = ds.X[df.loc[ds.X.index, target_prop] == False].index
    actives_test = ds.X_ind[df.loc[ds.X_ind.index, target_prop] == True].index
    inactives_test = ds.X_ind[df.loc[ds.X_ind.index, target_prop] == False].index
    # create temporary files
    active_train_file = save_temp(ds, actives_train)
    inactive_train_file = save_temp(ds, inactives_train)
    active_test_file = save_temp(ds, actives_test)
    inactive_test_file = save_temp(ds, inactives_test)
    # run script
    result_file = f"temp/{get_random_string()}.result"
    result = subprocess.run([
        sys.executable,
        "analyze_AVE_bias.py",
        "-activeMolsTraining",
        active_train_file,
        "-inactiveMolsTraining",
        inactive_train_file,
        "-activeMolsTesting",
        active_test_file,
        "-inactiveMolsTesting",
        inactive_test_file,
        "-outFile",
        result_file,
        "-fpType",
        "ECFP6",
    ], capture_output=True, text=True)
    # remove temporary files
    os.remove(active_train_file)
    os.remove(inactive_train_file)
    os.remove(active_test_file)
    os.remove(inactive_test_file)
    # parse results
    ret = dict()
    with open(result_file, "r") as f:
        result = f.read().strip()
        vals = [x for x in result.split("#") if x != ""]
        for val in vals:
            a, b = val.split("=")
            ret[a] = [float(b)]
    os.remove(result_file)
    # return
    return pd.DataFrame(ret)
