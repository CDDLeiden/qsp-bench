import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock

from qsprpred.models.tasks import TargetTasks
from sklearn.ensemble import ExtraTreesClassifier
from tools import get_dataset, benchmark_splits
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

# set False to avoid repeating experiments, will  only run the analysis part
rerun = False
# set to True if you want to fetch data sets again
reload_data = False

# load settings
grid = json.load(open("settings.json"))
model = ExtraTreesClassifier
model_params = {"n_estimators": 250}
endpoint = "pchembl_value_Median"
# FIXME: threshold very arbitrary here and might not be proper for al data sets
target_props = [{"name": endpoint, "task": TargetTasks.SINGLECLASS, 'th': [7]}]
seed = 42
n_proc = 20

lock = Lock()


def run_replica(replica, settings):
    with lock:
        ds = get_dataset(settings, target_props, reload=reload_data)
    seed_replica = seed + replica
    summary, bias_summary = benchmark_splits(ds, grid, model, model_params, seed=seed_replica, rerun=rerun)
    summary["Replica"] = replica
    summary["Seed"] = seed_replica
    bias_summary["Replica"] = replica
    bias_summary["Seed"] = seed_replica
    return summary, bias_summary


datasets = []
for setting in tqdm(data_settings):
    if rerun:
        results = None
        results_bias = None
        # loop over replicas in parallel
        with ProcessPoolExecutor(max_workers=n_proc) as executor:
            for sm, bs in executor.map(run_replica, range(n_replicas), [setting] * n_replicas):
                if results is None:
                    results = sm
                else:
                    results = pd.concat([results, sm])
                if results_bias is None:
                    results_bias = bs
                else:
                    results_bias = pd.concat([results_bias, bs])
        # save results
        results.to_csv(f"{setting['dataset_name']}_results.tsv", sep="\t", index=False, header=True)
        results_bias.to_csv(f"{setting['dataset_name']}_results_bias.tsv", sep="\t", index=False, header=True)
    else:
        results = pd.read_table(f"{setting['dataset_name']}_results.tsv")
        results_bias = pd.read_table(f"{setting['dataset_name']}_results_bias.tsv")

    # plot model performance
    df_ind = results.loc[(results.TestSet == "IND") & (results.Metric == "matthews_corrcoef")]
    plt.ylim([0, 1])
    plt.title(setting['dataset_name'])
    sns.boxplot(
        data=df_ind,
        x="gamma",
        y="Value",
        hue="sim_threshold",
        palette=sns.color_palette('bright')
    )
    plt.savefig(f"{setting['dataset_name']}_results.png")
    plt.clf()
    plt.close()

    # plot bias data
    for value in [
        "knn1",
        "lr",
        "rf",
        "svm",
        "AA-AI",
        "II-IA",
        "(AA-AI)+(II-IA)"
    ]:
        plt.ylim([0, 1])
        if value in ["AA-AI", "II-IA", "(AA-AI)+(II-IA)"]:
            plt.ylim([0, 0.5])
        plt.title(setting['dataset_name'])
        sns.boxplot(
            data=results_bias,
            x="gamma",
            y=value,
            hue="sim_threshold",
            palette=sns.color_palette('bright')
        )
        plt.savefig(f"{setting['dataset_name']}_results_bias_{value}.png")
        plt.clf()
        plt.close()
