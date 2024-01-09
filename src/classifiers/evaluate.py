import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score
)
from src.plotting import colorFader


def get_metrics(labels, predictions, cutoffs=[x/100 for x in range(1,100)], use_Youdens_cutoff=True):
    '''
    Return a range of metrics for a given set of labels and predictions

    :param labels: ndarray (N,) of binary labels
    :param predictions: ndarray (N,1) of probabilities
    :param cutoffs: list of cutoffs to try for binary predictions
    :param use_Youdens_cutoff: bool using cut-off that maximises Youden index
    :returns: df of metrics
    '''
    # binary predictions
    cutoff = get_optimal_Youden_cutoff(labels, predictions, cutoffs) if use_Youdens_cutoff else 0.5
    binary_predictions = [0 if pred < cutoff else 1 for pred in predictions]

    # metrics
    precision, recall, _ = precision_recall_curve(labels, predictions)
    PR_AUC = auc(recall, precision)
    F_score = f1_score(labels, binary_predictions)
    MCC = matthews_corrcoef(labels, binary_predictions)
    ROC_AUC = roc_auc_score(labels, predictions)
    bal_acc = balanced_accuracy_score(labels, binary_predictions)

    # format nicely
    metrics_arr = np.array([[PR_AUC, F_score, MCC, ROC_AUC, bal_acc, cutoff]])
    metrics_df = pd.DataFrame(metrics_arr, columns=["PR AUC", "F-score", "MCC", "ROC AUC", "Balanced Accuracy", "Youden cutoff"])
    metrics_df = metrics_df.round(3)
    
    return metrics_df


def get_Youden_index(labels, predictions, cutoff):
    '''
    Calculate Youden index for a given set of labels and predictions at a given cutoff
    :param labels: ndarray (N,) of binary labels
    :param predictions: ndarray (N,1) of probabilities
    :param cutoff: float cutoff for binary predictions
    :returns: float Youden index
    '''
    # setting adjusted=True is equivalent to Youdens calculation
    # https://scikit-learn.org/stable/modules/model_evaluation.html
    binary_predictions = [0 if pred < cutoff else 1 for pred in predictions]
    return balanced_accuracy_score(labels, binary_predictions, adjusted=True)    


def get_optimal_Youden_cutoff(labels, predictions, cutoffs):
    '''
    Find the cutoff that maximises the Youden index

    :param labels: ndarray (N,) of binary labels
    :param predictions: ndarray (N,1) of probabilities
    :param cutoffs: list of cutoffs to try for binary predictions
    '''
    Youden_values = []
    for _, cutoff in enumerate(cutoffs):
        Youden_values.append(get_Youden_index(labels, predictions, cutoff))
        
    max_J = max(Youden_values)
    max_J_idx = Youden_values.index(max_J)
    return cutoffs[max_J_idx]


def plot_PR_AUC(labels, predictions, title="", figsize=(4,4)):
    '''
    Plot precision-recall curve

    :param labels: ndarray (N,) of binary labels
    :param predictions: ndarray (N,1) of probabilities
    :param title: str title to put at top of plot
    :param figsize: tuple figsize of plot
    '''
    plt.rcParams["figure.facecolor"] = "white"
    fig, ax = plt.subplots(figsize=figsize)
    precision, recall, _ = precision_recall_curve(labels, predictions)
    plt.plot(recall, precision)

    plt.ylim([-0.05,1.05])
    plt.xlim([-0.05,1.05])
    ax.set_aspect("equal", "box")

    plt.xlabel("recall", fontsize=14)
    plt.ylabel("precision", fontsize=14)
    plt.title(title, fontsize=14)
    plt.show()


def plot_PR_AUC_by_edit_distance(labels, predictions, df, label_col="label",
                                 edit_distance_col="edit_distance", title="", figsize=(4,4),
                                 dpi=100, fontsize=None, colour_start_end=None):
    '''
    Plot precision-recall curve for each edit distance separately

    :param labels: ndarray (N,) of binary labels
    :param predictions: ndarray (N,1) of probabilities
    :param df: df containing edit distance info
    :param label_col: str name of col with binary labels
    :param edit_distance_col: str name of col with edit distances
    :param title: str title to put at top of plot
    :param figsize: tuple figsize of plot
    :param dpi: int dpi of plot
    :param fontsize: int fontsize for plot text
    :param colour_start_end: tuple of str colours to fade between for each edit distance
    '''
    plt.rcParams["figure.facecolor"] = "white"
    labels = labels.tolist()
    assert labels == df[label_col].tolist(), "df, labels, and preds should be in the same order"
    _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    edit_distances = df[edit_distance_col].unique().tolist()
    edit_distances.sort()
    df["predictions"] = predictions

    for edit_distance in edit_distances:
        y = df[df[edit_distance_col]==edit_distance][label_col].tolist()
        p = df[df[edit_distance_col]==edit_distance]["predictions"].tolist()
        precision, recall, _ = precision_recall_curve(y, p)
        pr_auc = auc(recall, precision)
        if colour_start_end is None:
            plt.plot(recall, precision, label=f"{edit_distance} - {pr_auc:.2f}")
        else:
            c1, c2 = colour_start_end
            plt.plot(recall, precision, label=f"{edit_distance} - {pr_auc:.2f}",
                     color=colorFader(c1,c2,edit_distance/max(edit_distances)))

    plt.ylim([-0.05,1.05])
    plt.xlim([-0.05,1.05])
    ax.set_aspect("equal", "box")

    plt.xlabel("Recall", fontsize=fontsize)
    plt.ylabel("Precision", fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.legend(fontsize=fontsize)

    plt.show()


def plot_predictions_histogram(labels, predictions, ymax=None, num_bins=20, title="", figsize=(4,2),
                               dpi=100, fontsize=12, xlabel="prediction", ylabel="", yvisible=True,
                               xvisible=True, legend=True, palette={1:"tab:orange", 0:"tab:blue", "na":"grey"}):
    '''
    Plot histogram of predictions, separated by label

    :param labels: ndarray (N,) of binary labels
    :param predictions: ndarray (N,1) of probabilities
    :param ymax: int max y value for plot
    :param num_bins: int number of bins for histogram
    :param title: str title to put at top of plot
    :param figsize: tuple figsize of plot
    :param dpi: int dpi of plot
    :param fontsize: int fontsize for plot text
    :param xlabel: str label for x axis
    :param ylabel: str label for y axis
    :param yvisible: bool whether to show y axis
    :param xvisible: bool whether to show x axis
    :param legend: bool whether to show legend
    :param palette: dict of colours for each label
    '''
    plt.rcParams["figure.facecolor"] = "white"
    df_pred = pd.DataFrame(list(zip(labels, predictions.flatten())), columns =["label", "prediction"])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # remove y axis
    if not yvisible:
        ax.axes.get_yaxis().set_visible(False)
    # remove x axis
    if not xvisible:
        ax.axes.get_xaxis().set_visible(False)
    # show only 0 and 1 on x axis
    plt.xticks([0,1])

    sns.histplot(df_pred, x="prediction", hue="label", bins=num_bins, binrange=(0,1), palette=palette)
    plt.title(title, fontsize=14)
    plt.xlim([0,1])
    plt.ylim([0, ymax]) if ymax else plt.ylim(0)
    # plt.legend(fontsize=fontsize)
    if not legend:
        plt.legend().remove()
    plt.show()


def plot_predictions_histogram_stack_editdist(df_predictions_editdist, ymax=None,
                                              pred_col="predictions", edit_col="edit_distance",
                                              num_bins=20, title="", figsize=(5,3), dpi=100,
                                              palette=None, xlabel="prediction", ylabel="count",
                                              fontsize=14, bbox_to_anchor=(1.05, 0.5), leg_title="Edit Dist.",
                                              vertical_line_at=None, vertical_line_text="",
                                              vertical_line_ymax=0.85):
    '''
    Plot histogram of predictions, separated by edit distance

    :param df_predictions_editdist: df containing edit distances and prediction values
    :param ymax: int max y value for plot
    :param pred_col: str name of col with predictions
    :param edit_col: str name of col with edit distances
    :param num_bins: int number of bins for histogram
    :param title: str title to put at top of plot
    :param figsize: tuple figsize of plot
    :param dpi: int dpi of plot
    :param palette: dict of colours for each label
    :param xlabel: str label for x axis
    :param ylabel: str label for y axis
    :param fontsize: int fontsize for plot text
    :param bbox_to_anchor: tuple of floats for legend position
    :param leg_title: str title for legend
    :param vertical_line_at: float x value for vertical line to highlight possible cutoff value
    :param vertical_line_text: str text to put at top of vertical line
    :param vertical_line_ymax: float ymax for vertical line
    '''
    _, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # plot data
    if palette is None:
        palette = {n: col for n, col in zip([i for i in range(1,11)], sns.color_palette("hls", 10))}
    sns.histplot(data=df_predictions_editdist,
                bins=num_bins,
                binrange=(0,1),
                x=pred_col,
                hue=edit_col,
                multiple="stack",
                palette=palette)

    # legend formatting
    sns.move_legend(ax, "center left", bbox_to_anchor=bbox_to_anchor,
                    title=leg_title, title_fontsize=fontsize, fontsize=fontsize)

    # axes formatting
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlim([0,1])
    plt.ylim([0, ymax]) if ymax else plt.ylim(0)
    plt.title(title, fontsize=fontsize)

    # vertical line
    if vertical_line_at is not None:
        plt.axvline(x=vertical_line_at, color="black", linestyle="--", ymax=vertical_line_ymax)
        plt.text(vertical_line_at, plt.ylim()[1]*0.9, vertical_line_text, fontsize=fontsize,
                 horizontalalignment='center')

    plt.show()

