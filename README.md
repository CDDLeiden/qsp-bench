# Various Benchmarking Experiments Implemented in QSPRpred

This repository contains the code used to conduct the case study experiments in the paper titled "QSPRpred: a Flexible Open-Source Quantitative Structure-Property Relationship Modelling Tool" and is intended as a reference implementation of a streamlined QSPR modelling worfklow for the purpose of comparison of models and other aspects of QSPR modelling.

## Install Dependencies

Different dependencies are required for different experiments. The following should be sufficient for all experiments:

```bash
pip install deepchem # for MoleculeNet data sets
pip install qsprpred[gpu] # for software dependencies
```

## Run Experiments

Settings for each experiment is stored as a module under `settings`. You just point to the module you wish to run with the `QSPBENCH_SETTINGS` environment variable and run the `main.py` script. For example, to run the multi-task regression experiments from the first experiment you can issue the following command:

```bash
QSPBENCH_SETTINGS='settings.experiment_1.multitask_reg' python main.py
```

Some settings files define other variables that can be used to customize behaviour. Check the settings you intend to run for more details.

## Analyze Results

For each experiment, a Jupyter notebook that reads and analyzes the output data is available in the root of the repository (i.e. [`analyze_experiment_1.ipynb`](analyze_experiment_1.ipynb)).
