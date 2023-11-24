from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock

from benchmark_settings import ReplicaSettings
from settings import SETTINGS, N_PROC
from tools import get_dataset, benchmark_splits
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

# set False to avoid repeating experiments, will  only run the analysis part
rerun = False
# set to True if you want to fetch data sets again
reload_data = False

lock = Lock()


def run_replica(replica: ReplicaSettings):
    with lock:
        ds = get_dataset(replica, reload=reload_data)
    seed_replica = seed + replica
    summary, bias_summary = benchmark_splits(ds, grid, model, model_params, seed=seed_replica, rerun=rerun)
    summary["Replica"] = replica
    summary["Seed"] = seed_replica
    bias_summary["Replica"] = replica
    bias_summary["Seed"] = seed_replica
    return summary, bias_summary


datasets = []
if rerun:
    results = None
    results_bias = None
    # loop over replicas in parallel
    with ProcessPoolExecutor(max_workers=N_PROC) as executor:
        for sm, bs in executor.map(run_replica, SETTINGS.iter_replicas()):
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
