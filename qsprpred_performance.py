import logging

import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Lock

from benchmark_settings import Replica
from settings import SETTINGS, N_PROC, RESULTS_FILE, DATA_DIR
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


lock = Lock()
def run_replica(replica: Replica):
    try:
        with lock:
            df_results = None
            if os.path.exists(RESULTS_FILE):
                df_results = pd.read_table(RESULTS_FILE)
            if df_results is not None and df_results.ReplicaID.isin([replica.id]).any():
                logging.warning(f"Skipping {replica.id}")
                return
            replica.create_dataset(reload=False)
        replica.prep_dataset()
        df_report = replica.create_report()
        with lock:
            df_report.to_csv(
                RESULTS_FILE,
                sep="\t",
                index=False,
                mode="a",
                header=not os.path.exists(RESULTS_FILE)
            )
            logging.info(f"Finished {replica.id}.")
    except Exception as e:
        logging.error(f"Error in {replica.id}:")
        logging.exception(e)
        return replica.id, e


results = None
# results_bias = None
# loop over replicas in parallel
with ProcessPoolExecutor(max_workers=N_PROC) as executor:
    for model_summary in executor.map(
            run_replica,
            SETTINGS.iter_replicas()
    ):
        if model_summary is not None:
            logging.error("Something went wrong: ", model_summary[1])


results = pd.read_table(RESULTS_FILE)
# plot model performance
for score_func in results.ScoreFunc.unique():
    df_ind = results.loc[(results.ScoreFunc == score_func)]
    plt.ylim([0, 1])
    plt.title(score_func)
    sns.boxplot(
        data=df_ind,
        x="DataSet",
        y="Score",
        hue="Algorithm",
        palette=sns.color_palette('bright')
    )
    plt.savefig(f"{DATA_DIR}/{score_func}_results.png")
    plt.clf()
    plt.close()

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
