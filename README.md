# Various Benchmarking Experiments Implemented in QSPRpred

## Running Experiments

### Install Dependencies

Different dependencies are required for different experiments. The following should be sufficient for all experiments:

```bash
conda install cudatoolkit=11.1
pip install -r requirements.txt
pip install qsprpred==3.0.0[pyboost]
```

### Run Experiments

Settings for each experiment is stored as a module under `settings`. You just point to the module you wish to run with the `QSPBENCH_SETTINGS` environment variable and run the `main.py` script. For example, to run the multi-task regression experiments you can issue the following command:

```bash
QSPBENCH_SETTINGS='settings.multitask_reg' python main.py
```