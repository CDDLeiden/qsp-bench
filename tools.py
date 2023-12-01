import os
import subprocess
import sys

import pandas as pd
from qsprpred.data.data import MoleculeTable
from qsprpred.data.sources.papyrus import Papyrus
from qsprpred.utils.stringops import get_random_string


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
