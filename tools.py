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
