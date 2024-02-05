import pandas as pd

from qsprpred import TargetProperty
from qsprpred.data import MoleculeTable, QSPRDataset
from qsprpred.data.sources.papyrus import Papyrus
from qsprpred.logs import logger


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
            n_samples = min(self.nSamples, len(ret))
            return ret.sample(n_samples, name)

    def getDataSet(
        self,
        target_props: list[TargetProperty | dict],
        name: str | None = None,
        **kwargs
    ) -> QSPRDataset:
        # kwargs["store_format"] = "csv"
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
                df_acc = df[df["accession"] == acc]
                dfs_sample.append(df_acc.sample(min(n_samples, len(df_acc))))
            df = pd.concat(dfs_sample)
            logger.info(f"Sampled {len(df)} molecules for {name}")
        df = df.pivot(index="SMILES", columns="accession", values="pchembl_value_Mean")
        df.columns.name = None
        df.reset_index(inplace=True)
        return MoleculeTable(
            name=prior.name,
            df=df,
            store_dir=self.dataDir,
            **kwargs
        )
