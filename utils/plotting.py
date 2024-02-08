from matplotlib import pyplot as plt
import seaborn as sns


def make_box_plot(data, x, y, hue, plot_name="boxplot", out_dir="./data/", ylim=(0, 1)):
    """
    Make a box plot with given settings.

    Args:
        data (pd.DataFrame): Data to plot.
        x (str): Column name for x-axis.
        y (str): Column name for y-axis.
        hue (str): Column name for hue.
        plot_name (str): Name of plot.
        out_dir (str): Directory to save plot to.
        ylim (tuple): Y-axis limits.
    """

    for score_func in data.ScoreFunc.unique():
        df_ind = data.loc[(data.ScoreFunc == score_func)]
        plt.ylim(ylim)
        plt.title(plot_name + " " + score_func)
        sns.boxplot(
            data=df_ind,
            x=x,
            y=y,
            hue=hue,
            # palette=sns.color_palette('bright')
        )
        plt.savefig(f"{out_dir}/{plot_name}_{score_func}_{x}_{y}_{hue}.png")
        plt.clf()
        plt.close()
