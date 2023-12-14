from sklearn.impute import SimpleImputer

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
            plus_only=True
        )
        self.accKeys = sorted(acc_keys)
        self.nSamples = n_samples

    def getData(
        self,
        name: str | None = None,
        **kwargs,
    ) -> MoleculeTable:
        return super().getData(
            name=name or "_".join(self.accKeys),
            acc_keys=self.accKeys,
            quality="high",
            activity_types="all",
            drop_duplicates=True,
            use_existing=True,
            **kwargs,
        ).sample(self.nSamples)


class PapyrusForBenchmarkMT(PapyrusForBenchmark):

    def getData(
        self,
        name: str | None = None,
        **kwargs,
    ) -> MoleculeTable:
        prior = super().getData(
            name=name or "_".join(self.accKeys),
            acc_keys=self.accKeys,
            quality="high",
            activity_types="all",
            drop_duplicates=True,
            use_existing=True,
            **kwargs,
        ).sample(self.nSamples)
        df = prior.getDF()
        df = df.pivot(index="SMILES", columns="accession", values="pchembl_value_Mean")
        df.columns.name = None
        df.reset_index(inplace=True)
        return MoleculeTable(
            name=prior.name,
            df=df,
            **kwargs
        )

    def getDataSet(
        self,
        target_props: list[TargetProperty | dict],
        name: str | None = None,
        **kwargs
    ) -> QSPRDataset:
        kwargs["target_imputer"] = SimpleImputer(strategy="mean")
        return super().getDataSet(
            target_props=target_props,
            name=name,
            **kwargs
        )
