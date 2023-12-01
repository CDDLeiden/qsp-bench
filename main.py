from qsprpred.benchmarks import BenchmarkRunner
from settings import SETTINGS, N_PROC, RESULTS_FILE, DATA_DIR
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

runner = BenchmarkRunner(SETTINGS, N_PROC, DATA_DIR, RESULTS_FILE)
results = runner.run()
results.DataSet = results.DataSet.apply(lambda x: x.split("_")[0])
results.Algorithm = results.Algorithm.apply(lambda x: x.split(".")[-1])
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
