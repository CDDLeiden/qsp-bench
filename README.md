# Various Benchmarking Experiments Implemented in QSPRpred

## Running Experiments

### Install Dependencies

Different dependencies are required for different experiments. The following should be sufficient for all experiments:

```bash
pip install deepchem # for MoleculeNet data sets
pip install qsprpred[gpu] # for software dependencies
```

### Run Experiments

Settings for each experiment is stored as a module under `settings`. You just point to the module you wish to run with the `QSPBENCH_SETTINGS` environment variable and run the `main.py` script. For example, to run the multi-task regression experiments you can issue the following command:

```bash
QSPBENCH_SETTINGS='settings.multitask_reg' python main.py
```

Some settings files define other variables that can be used to customize behaviour. Check the settings you intend to run for more details.
