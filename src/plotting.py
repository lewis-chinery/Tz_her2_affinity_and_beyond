import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
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


def get_method_colour_map ():
    return {"trastuzumab": "black",
            "her2_aff_lrg": "grey",
            "blosum": "green",
            "random": "limegreen",
            "ablang_all": "darkviolet",
            "ablang_one": "indigo",
            "esm_all": "darkblue",
            "esm_one": "blue",
            "protein_mpnn": "firebrick"}


def make_nice_tsne_axes(fontsize=16, figsize=(7,7)):
    '''
    Make nice axes for tsne plots

    :param fontsize: fontsize for axes labels
    :param figsize: figsize for plot
    :return: fig, ax
    '''
    fig, ax = plt.subplots(figsize=figsize)
    plt.rcParams.update({"font.size": fontsize,
                         "font.family": "monospace",
                         "text.color" : "black",
                         "axes.labelcolor" : "black",
                         "figure.facecolor":  "white",
                         "axes.facecolor":    "white",
                         "savefig.facecolor": "white"
                         })
    plt.xticks([])
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.xlabel("t-SNE 1", fontsize=fontsize)
    plt.ylabel("t-SNE 2", fontsize=fontsize)
    return fig, ax


def make_df_for_tsne_plotting(z, y, to_test=None):
    '''
    Make df for tsne plotting

    :param z: tsne output
    :param y: method labels
    :param to_test: binary list whether or not to test (can be used for different sized markers later)
    :return: plotting_df
    '''
    plotting_df = pd.DataFrame()
    plotting_df["y"] = y
    plotting_df["comp-1"] = z[:,0]
    plotting_df["comp-2"] = z[:,1]
    if to_test == None:
        to_test = np.array([0] * len(y))
    plotting_df["to_test"] = to_test
    return plotting_df


def make_nice_legend_formatting_of_methods(methods):
    '''
    Make nice legend formatting of methods

    :param methods: list of methods that are normally used in analysis
    :return: list of methods in same order but made pretty
    '''
    new_methods = []
    for method in methods:
        if method == "her2_aff_lrg": new_methods.append("HER2-aff-lrg")
        if method == "blosum": new_methods.append("BLOSUM")
        if method == "ablang_all": new_methods.append("AbL (all)")
        if method == "ablang_one": new_methods.append("AbL (one)")
        if method == "esm_one": new_methods.append("ESM (one)")
        if method == "esm_all": new_methods.append("ESM (all)")
        if method == "protein_mpnn": new_methods.append("MPNN")
    return new_methods


def plot_tsne(z, y, to_test=None, fontsize=16, figsize=(7,7), dms_alt_palette=None,
              method_colour_map=None, markers={0: ".", 1: "X"}, sizes={0: 10, 1: 30},
              trastuzumab_size=200):
    '''
    Plot tsne

    :param z: tsne output
    :param y: method labels
    :param to_test: binary list whether or not to test (can be used for different sized markers)
    :param fontsize: fontsize for axes labels
    :param figsize: figsize for plot
    :param dms_alt_palette: list of colours to use for dms alternatives
    :param method_colour_map: dict of method names to colours
    :param markers: dict of what shape marker to do for test (1)/not test (0)
    :param sizes: dict of what size marker to do for test (1)/not test (0)
    :param trastuzumab_size: size of trastuzumab marker
    '''
    _, ax = make_nice_tsne_axes(fontsize=fontsize, figsize=figsize)
    plotting_df = make_df_for_tsne_plotting(z, y, to_test=to_test)
    method_colour_map = get_method_colour_map() if method_colour_map == None else method_colour_map

    # dms alternatives
    sns.scatterplot(x="comp-1",
                    y="comp-2",
                    hue="y",
                    style="to_test",
                    markers=markers,
                    size="to_test",
                    sizes=sizes,
                    palette=dms_alt_palette,
                    data=plotting_df[plotting_df["y"] != "trastuzumab"],
                    ax=ax)

    # trastuzumab
    sns.scatterplot(x="comp-1",
                    y="comp-2",
                    hue="y",
                    palette=[method_colour_map["trastuzumab"]],
                    data=plotting_df[plotting_df["y"] == "trastuzumab"],
                    s=trastuzumab_size,
                    ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    labels = labels[1:len(np.unique(y))]
    labels = make_nice_legend_formatting_of_methods(labels)
    handles = handles[1:]
    ax.legend(handles=handles, labels=labels, facecolor="inherit",
              bbox_to_anchor=(1,0.70), fontsize=14, title="")
    print(handles)
    plt.show()
