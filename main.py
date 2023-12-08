import seaborn as sns
import matplotlib.pyplot as plt

from qsprpred.benchmarks import BenchmarkRunner
from common_settings import SETTINGS, N_PROC, DATA_DIR, RESULTS_FILE
from multitask_settings import clustersplit_settings, randomsplit_settings

settings = SETTINGS
#settings = { **SETTINGS, **clustersplit_settings() }
#settings = { **SETTINGS, **randomsplit_settings() }

runner = BenchmarkRunner(settings, N_PROC, DATA_DIR, RESULTS_FILE)
results = runner.run(raise_errors=True)
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

