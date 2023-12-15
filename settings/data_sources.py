from qsprpred.data import MoleculeTable
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
        return super().getData(
            name=name,
            acc_keys=self.accKeys,
            quality="high",
            activity_types="all",
            drop_duplicates=False,
            use_existing=True,
            output_dir=self.dataDir,
            **kwargs,
        ).sample(self.nSamples, name)


class PapyrusForBenchmarkMT(PapyrusForBenchmark):

    def getData(
        self,
        name: str | None = None,
        **kwargs,
    ) -> MoleculeTable:
        prior = super().getData(
            name=name,
            **kwargs,
        )
        df = prior.getDF()
        df = df.pivot(index="SMILES", columns="accession", values="pchembl_value_Mean")
        df.columns.name = None
        df.reset_index(inplace=True)
        return MoleculeTable(
            name=prior.name,
            df=df,
            store_dir=self.dataDir,
            **kwargs
        )
