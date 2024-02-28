#!/bin/bash

# function to run a single experiment from given parameters
function run_experiment {
    local QSPBENCH_GPUS=$1
    local QSPBENCH_MOLNETSET=$2
    local QSPBENCH_SETTINGS=$3
    local QSPRPRED_VERBOSE_LOGGING=$4
    local QSPBENCH_NPROC=$5

    echo "Running experiment with parameters:"
    echo "QSPBENCH_GPUS=$QSPBENCH_GPUS"
    echo "QSPBENCH_MOLNETSET=$QSPBENCH_MOLNETSET"
    echo "QSPBENCH_SETTINGS=$QSPBENCH_SETTINGS"
    echo "QSPRPRED_VERBOSE_LOGGING=$QSPRPRED_VERBOSE_LOGGING"

    QSPBENCH_GPUS=$QSPBENCH_GPUS QSPBENCH_MOLNETSET=$QSPBENCH_MOLNETSET QSPBENCH_SETTINGS=$QSPBENCH_SETTINGS QSPRPRED_VERBOSE_LOGGING=$QSPRPRED_VERBOSE_LOGGING QSPBENCH_NPROC=${QSPBENCH_NPROC} ./run_experiment.sh
}

GPUS="1,2,3,4,5,6,7"
CPUS="12"
# Experiment 1: Regression with Chemprop
run_experiment $GPUS "Lipophilicity" "settings.experiment_2.regression_chemprop" "true" $CPUS
run_experiment $GPUS "delaney-processed" "settings.experiment_2.regression_chemprop" "true" $CPUS
run_experiment $GPUS "freesolv" "settings.experiment_2.regression_chemprop" "true" $CPUS
run_experiment $GPUS "clearance" "settings.experiment_2.regression_chemprop" "true" $CPUS

# Experiment 1: Regression with DNN
run_experiment $GPUS "Lipophilicity" "settings.experiment_2.regression_dnn" "true" $CPUS
run_experiment $GPUS "delaney-processed" "settings.experiment_2.regression_dnn" "true" $CPUS
run_experiment $GPUS "freesolv" "settings.experiment_2.regression_dnn" "true" $CPUS
run_experiment $GPUS "clearance" "settings.experiment_2.regression_dnn" "true" $CPUS

# Experiment 2: Classification with Chemprop
run_experiment $GPUS "HIV" "settings.experiment_2.classification_chemprop" "true" $CPUS
run_experiment $GPUS "bace" "settings.experiment_2.classification_chemprop" "true" $CPUS
run_experiment $GPUS "BBBP" "settings.experiment_2.classification_chemprop" "true" $CPUS

# Experiment 2: Classification with DNN
run_experiment $GPUS "HIV" "settings.experiment_2.classification_dnn" "true" $CPUS
run_experiment $GPUS "bace" "settings.experiment_2.classification_dnn" "true" $CPUS
run_experiment $GPUS "BBBP" "settings.experiment_2.classification_dnn" "true" $CPUS
