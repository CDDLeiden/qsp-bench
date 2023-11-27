import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock

from benchmark_settings import Replica
from settings import SETTINGS, N_PROC
from tools import get_dataset, benchmark_replica, get_AVE_bias
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# set False to avoid repeating experiments, will  only run the analysis part
rerun = False
# set to True if you want to fetch data sets again
reload_data = False

lock = Lock()


def run_replica(replica: Replica):
    with lock:
        ds = get_dataset(replica, reload=reload_data)
    if rerun or not replica.is_finished:
        model = benchmark_replica(ds, replica)
        replica.model = model
        replica.is_finished = True
    return replica, get_AVE_bias(ds, ds.targetProperties[0].name)


results = None
results_bias = None
# loop over replicas in parallel
with ProcessPoolExecutor(max_workers=N_PROC) as executor:
    for replica, bias_summary in executor.map(
            run_replica,
            SETTINGS.iter_replicas()
    ):
        replica = pd.DataFrame(
            data={
                "ID": replica.id,
                "ModelName": replica.model.name,
                "Replica": replica.toJSON(),
            }
        )
        if results is None:
            results = replica
        else:
            results = pd.concat([results, replica])
        if results_bias is None:
            results_bias = bias_summary
        else:
            results_bias = pd.concat([results_bias, bias_summary])
# save results
results.to_csv(
    "results.tsv",
    sep="\t",
    index=False,
    mode="a",
    header=not os.path.exists("results.tsv")
)
results_bias.to_csv(
    "results_bias.tsv",
    sep="\t",
    index=False,
    mode="a",
    header=not os.path.exists("results_bias.tsv")
)

# plot model performance
# df_ind = results.loc[(results.TestSet == "IND") & (results.Metric == "matthews_corrcoef")]
# plt.ylim([0, 1])
# plt.title(setting['dataset_name'])
# sns.boxplot(
#     data=df_ind,
#     x="gamma",
#     y="Value",
#     hue="sim_threshold",
#     palette=sns.color_palette('bright')
# )
# plt.savefig(f"{setting['dataset_name']}_results.png")
# plt.clf()
# plt.close()

# plot bias data
# for value in [
#     "knn1",
#     "lr",
#     "rf",
#     "svm",
#     "AA-AI",
#     "II-IA",
#     "(AA-AI)+(II-IA)"
# ]:
#     plt.ylim([0, 1])
#     if value in ["AA-AI", "II-IA", "(AA-AI)+(II-IA)"]:
#         plt.ylim([0, 0.5])
#     plt.title(setting['dataset_name'])
#     sns.boxplot(
#         data=results_bias,
#         x="gamma",
#         y=value,
#         hue="sim_threshold",
#         palette=sns.color_palette('bright')
#     )
#     plt.savefig(f"{setting['dataset_name']}_results_bias_{value}.png")
#     plt.clf()
#     plt.close()
