from matplotlib import pyplot as plt
import seaborn as sns


def make_box_plot(data, x, y, hue, plot_name="boxplot", out_dir="./data/", ylim=(0, 1), figsize=(8, 5), colors = [
        "#288AB5",
        "#0E5655",
        "#637E54",
        "#D8D1A9",
        "#E1AB24",
        "#DA2715",
        "#9A0320",
    ]):
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
    # make into  seaborn palette
    sns.set_palette(sns.color_palette(colors))
    for score_func in data.ScoreFuncNice.unique():
        df_ind = data.loc[(data.ScoreFuncNice == score_func)]
        plt.subplots(figsize=figsize)
        plt.ylim(ylim)
        ax = sns.boxplot(
            data=df_ind,
            x=x,
            y=y,
            hue=hue,
        )
        ax.set_ylabel(score_func)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        plt.title(plot_name + " " + score_func)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{plot_name}_{score_func}_{x}_{y}_{hue}.png")
        plt.clf()
        plt.close()
