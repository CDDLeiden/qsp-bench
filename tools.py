import os
import random
import string
import subprocess
import sys
import warnings
from itertools import product

import pandas as pd
from matplotlib import pyplot as plt
from qsprpred.data.data import QSPRDataset
from qsprpred.data.descriptors.calculators import MoleculeDescriptorsCalculator
from qsprpred.data.descriptors.sets import FingerprintSet
from qsprpred.data.sources.papyrus import Papyrus
from qsprpred.models.sklearn import SklearnModel
from qsprpred.plotting.classification import MetricsPlot


def get_benchmark_dataset_from_papyrus(settings, reload=False):
    data_dir = "./data"
    settings_current = {
        "acc_keys": ["P4159"],
        "dataset_name": "CCR2_HUMAN",
        "quality": "high",
        "papyrus_version": '05.6',
        "plus_only": True,
        "activity_types": "all"
    }
    settings_current.update(settings)

    papyrus = Papyrus(
        data_dir=data_dir,
        version=settings_current["papyrus_version"],
        stereo=False,
        plus_only=settings_current["plus_only"],
        descriptors=None
    )

    return papyrus.getData(
        settings_current["acc_keys"],
        settings_current["quality"],
        name=settings_current["dataset_name"],
        use_existing=not reload,
        activity_types=settings["activity_types"]
    )


def get_dataset(settings, target_props, reload=False):
    ds = None
    try:
        ds = QSPRDataset(name=settings["dataset_name"], store_dir="./data", target_props=target_props)
        desc_calculator = MoleculeDescriptorsCalculator(
            desc_sets=[
                FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=2048)
            ]
        )
        ds.addDescriptors(desc_calculator, recalculate=False)
        ds.save()
    except ValueError:
        print(f"Data set for {settings['dataset_name']} not found. It will be created.")

    if reload or ds is None:
        # get data
        mt = get_benchmark_dataset_from_papyrus(settings, reload=reload)
        ds = QSPRDataset.fromMolTable(mt, target_props=target_props)
        ds.save()

    return ds


def separate_into_dictionary(string):
    """Separates a string of key-value pairs into a Python dictionary.

    Args:
        string: A string of key-value pairs, separated by commas.

    Returns:
        A Python dictionary.
    """

    dictionary = {}
    for key_value_pair in string.split("(")[1].split(")")[0].split(","):
        key, value = key_value_pair.split("=")
        try:
            dictionary[key] = [float(value)]
        except ValueError:
            dictionary[key] = [value]
    return dictionary


def get_random_string():
    return ''.join(random.choice(string.ascii_letters) for i in range(10))


def save_temp(ds, index):
    df = ds.getDF()
    os.makedirs("temp", exist_ok=True)
    name = get_random_string()
    name = f"temp/{name}_temp.smi"
    df.loc[index, [ds.smilesCol] + ds.indexCols].to_csv(name, index=False, header=False, sep=" ")
    return name


def get_AVE_bias(ds, target_prop):
    df = ds.getDF()
    actives_train = ds.X[df.loc[ds.X.index, target_prop] == True].index
    inactives_train = ds.X[df.loc[ds.X.index, target_prop] == False].index
    actives_test = ds.X_ind[df.loc[ds.X_ind.index, target_prop] == True].index
    inactives_test = ds.X_ind[df.loc[ds.X_ind.index, target_prop] == False].index
    # create temporary files
    active_train_file = save_temp(ds, actives_train)
    inactive_train_file = save_temp(ds, inactives_train)
    active_test_file = save_temp(ds, actives_test)
    inactive_test_file = save_temp(ds, inactives_test)
    # run script
    result_file = f"temp/{get_random_string()}.result"
    result = subprocess.run([
        sys.executable,
        "analyze_AVE_bias.py",
        "-activeMolsTraining",
        active_train_file,
        "-inactiveMolsTraining",
        inactive_train_file,
        "-activeMolsTesting",
        active_test_file,
        "-inactiveMolsTesting",
        inactive_test_file,
        "-outFile",
        result_file,
        "-fpType",
        "ECFP6",
    ], capture_output=True, text=True)
    # remove temporary files
    os.remove(active_train_file)
    os.remove(inactive_train_file)
    os.remove(active_test_file)
    os.remove(inactive_test_file)
    # parse results
    ret = dict()
    with open(result_file, "r") as f:
        result = f.read().strip()
        vals = [x for x in result.split("#") if x != ""]
        for val in vals:
            a, b = val.split("=")
            ret[a] = [float(b)]
    os.remove(result_file)
    # return
    return pd.DataFrame(ret)


def benchmark_splits(ds, split_grid, model_class, model_params, seed=None, rerun=False):
    models = []
    keys_sorted = sorted(split_grid.keys())
    grid_sorted = [split_grid[key] for key in keys_sorted]
    settings = [x for x in product(*grid_sorted)]
    bias_summary = pd.DataFrame()
    for combo in settings:
        combo = {key: value for key, value in zip(keys_sorted, combo)}
        combo_str = ",".join([f"{key}={value}" for key, value in combo.items()])
        bias_data = None
        if rerun:
            # run the split if requested
            split = AnalogueSplit(
                draw=False,
                random_seed=seed,
                **combo
            )
            try:
                ds.split(split, featurize=True)
                bias_data = get_AVE_bias(ds, ds.targetProperties[0].name)
            except Exception as exp:
                warnings.warn(f"Split with settings: \n\t" +
                              f"{combo_str}\n" +
                              f"has failed ({exp}). Model skipped.")
                continue

        # load model
        model = SklearnModel(
            name=f"{ds.name}-ExtraTrees({combo_str},seed={seed})",
            data=ds,
            alg=model_class,
            parameters=model_params,
            base_dir="."
        )
        if rerun:
            # evaluate on the new split
            model.evaluate()
            model.save()

        #  add model to the current list
        models.append(model)
        if bias_data is not None:
            bias_data["Model"] = model.name
            bias_summary = pd.concat([bias_summary, bias_data])
    if len(models) == 0:
        raise ValueError("No models were evaluated. Check your settings.")

    # not functional, just an example to visualize interactively all splits for one or more models
    # from scaffviz.clustering.manifold import TSNE
    # from scaffviz.depiction.plot import ModelPerformancePlot

    # plot = ModelPerformancePlot(
    #     TSNE(random_state=42),
    #     models,
    #     plot_type="splits",
    #     async_execution=True,
    #     ports=[x for x in range(8001, 8001 + len(models),  1)]
    # )
    # info = plot.make()
    # info

    # get the summary table
    plot = MetricsPlot(models)
    figures, summary = plot.make(save=False, show=False, property_name=ds.targetProperties[0].name, out_dir="./qspr")
    df_params = pd.concat(
        summary.apply(lambda x: pd.DataFrame(separate_into_dictionary(x.Model)), axis=1).to_list()
    )
    df_params_bias = pd.concat(
        bias_summary.apply(lambda x: pd.DataFrame(separate_into_dictionary(x.Model)), axis=1).to_list()
    )
    df_params.index = summary.index
    plt.close('all')
    return pd.concat([summary, df_params], axis=1), pd.concat([bias_summary, df_params_bias], axis=1)
