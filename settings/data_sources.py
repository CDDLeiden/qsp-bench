import pandas as pd

from qsprpred import TargetProperty
from qsprpred.data import MoleculeTable, QSPRDataset
from qsprpred.data.sources.papyrus import Papyrus


class PapyrusForBenchmark(Papyrus):

    def __init__(
            self,
            acc_keys: list[str],
            data_dir: str,
            n_samples: int | None = None,
    ):
        super().__init__(
            data_dir=data_dir,
            version="05.6",
            plus_only=True,
        )
        self.accKeys = sorted(acc_keys)
        self.nSamples = n_samples
        self.dataDir = data_dir

    def getData(
        self,
        name: str | None = None,
        **kwargs,
    ) -> MoleculeTable:
        name = name or "_".join(self.accKeys)
        ret = super().getData(
                name=name,
                acc_keys=self.accKeys,
                quality="high",
                activity_types="all",
                drop_duplicates=False,
                use_existing=True,
                output_dir=self.dataDir,
                **kwargs,
            )
        if self.nSamples is None:
            return ret
        else:
            return ret.sample(self.nSamples, name)

    def getDataSet(
        self,
        target_props: list[TargetProperty | dict],
        name: str | None = None,
        **kwargs
    ) -> QSPRDataset:
        kwargs["store_format"] = "csv"
        return super().getDataSet(
            target_props=target_props,
            name=name,
            **kwargs,
        )


class PapyrusForBenchmarkMT(PapyrusForBenchmark):

    def getData(
        self,
        name: str | None = None,
        **kwargs,
    ) -> MoleculeTable:
        n_samples = self.nSamples
        self.nSamples = None
        prior = super().getData(
            name=name,
            **kwargs,
        )
        df = prior.getDF()
        if n_samples is not None:
            dfs_sample = []
            for acc in self.accKeys:
                dfs_sample.append(df[df.accession == acc].sample(n_samples))
            df = pd.concat(dfs_sample)
        df = df.pivot(index="SMILES", columns="accession", values="pchembl_value_Mean")
        df.columns.name = None
        df.reset_index(inplace=True)
        return MoleculeTable(
            name=prior.name,
            df=df,
            store_dir=self.dataDir,
            **kwargs
        )
