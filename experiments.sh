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

CPUS="12"

# Experiment 2: Regression with other models
run_experiment '' "Lipophilicity" "settings.experiment_2.regression" "true" $CPUS
run_experiment '' "delaney-processed" "settings.experiment_2.regression" "true" $CPUS
run_experiment '' "freesolv" "settings.experiment_2.regression" "true" $CPUS
run_experiment '' "clearance" "settings.experiment_2.regression" "true" $CPUS

# Experiment 2: Classification with other models
run_experiment '' "HIV" "settings.experiment_2.classification" "true" $CPUS
run_experiment '' "bace" "settings.experiment_2.classification" "true" $CPUS
run_experiment '' "BBBP" "settings.experiment_2.classification" "true" $CPUS
