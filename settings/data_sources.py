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
