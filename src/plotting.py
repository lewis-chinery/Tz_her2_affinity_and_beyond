import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import logomaker as lm


def plot_logo_plot_from_normalised_matrix(normalised_mat, title="", xticks_old=[i for i in range(0,10)],
                                          xticks_new=[i for i in range(107,117)], yticks=[],
                                          figsize=(10,2), dpi=None, fontsize=14, rotation=0):
    '''
    Plot logo plot with AAs of different heights depending on their desired frequency
    :param normalised_mat: ndarray 2D, rows are positions (N), columns are AAs (20)
    '''
    fix, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.rcParams["figure.facecolor"] = "white"
    logo = lm.Logo(normalised_mat, color_scheme="chemistry", ax=ax)
    plt.xticks(xticks_old, xticks_new, fontsize=fontsize, rotation=rotation)
    plt.yticks(yticks, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.show()


def plot_edit_distances(df, positives_only=False, edit_distance_col="edit_distance",
    label_col="label", title="", figsize=(2,1.5), dpi=100, ymax=None, target_counts=None,
    fontsize=12):
    '''
    Plot edit distances observed in training data
    We wish to recreate this distribution with our generated libraries

    :param df: df containing edit distance info
    :param positives_only: bool True if only using positive labels
    :param edit_distance_col: str name of col with edit distances
    :param label_col: str name of col with binary labels
    :param title: str title to put at top of plot
    :param figsize: tuple figsize of plot
    '''
    plt.rcParams["figure.facecolor"] = "white"
    if positives_only:
        df = df[df[label_col]==1]

    _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(df[edit_distance_col],
            bins=round(df[edit_distance_col].max()+1),
            range=(-0.5, df[edit_distance_col].max()+0.5),
            color="tab:blue", alpha=0.5)
    
    if target_counts:
        # convert dictionary into count list with each key repeated by the value
        target_counts_list = []
        for k, v in target_counts.items():
            target_counts_list += [k]*v
        ax.hist(target_counts_list,
                bins=round(df[edit_distance_col].max()+1),
                range=(-0.5, df[edit_distance_col].max()+0.5),
                histtype="step",
                color="black",
                linewidth=.5,
                alpha=1)

    if ymax:
        plt.ylim([0, ymax])
    plt.xlabel("Edit dist.", fontsize=fontsize)
    plt.ylabel("Count", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.show()


def buzz_colours():
    '''
    Save colours for plots with Buzz theme

    :returns: dict of colours as hex codes
    '''
    dark_purple = "#9900FF"
    light_purple = "#CA7DFF"
    dark_green = "#00B050"
    light_green = "#92D050"
    yellowy_gold = "#FFC000"
    bright_red = "#FF0000"
    blue = "#6495ED"
    return {"dark_purple": dark_purple, "light_purple": light_purple,
            "dark_green": dark_green, "light_green": light_green,
            "yellowy_gold": yellowy_gold, "bright_red": bright_red,
            "blue": blue}


def colorFader(c1,c2,mix):
    '''
    https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
    fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)

    :param c1: str hex code of colour 1
    :param c2: str hex code of colour 2
    :param mix: float 0-1 distance between c1 and c2
    :returns: str hex code of new colour
    '''
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)
