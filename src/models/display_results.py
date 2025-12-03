import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import itertools
#import shap

__author__ = "Roude JEAN MARIE"
__email__ = "roude.bioinfo@gmail.com"

FIGSIZE = (10, 9)


def display_train_test_scores(
    train_scores, test_scores, figsize=FIGSIZE,
    hline=True, title="", bottom=-1, top=1,
    filepath=None, show=False
):
    """"""
    if isinstance(train_scores, dict):
        train_scores = pd.DataFrame(train_scores).T
    if isinstance(test_scores, dict):
        test_scores = pd.DataFrame(test_scores).T
    test_scores = test_scores.loc[train_scores.index, :]
    # 
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train_scores.index, train_scores.iloc[:, 0], label=f"train", marker="o")
    ax.plot(test_scores.index, test_scores.iloc[:, 0], label=f"test", marker="o")
    if hline:
        ax.axhline(0.5, color="g", linestyle="--")
        ax.axhline(0, color="b", linestyle="--")
    ax.set_ylim(bottom, top)
    ax.set_xticks([i for i in range(len(train_scores.index))])
    ax.set_xticklabels(
        train_scores.index, rotation=45, ha='right', rotation_mode='anchor'
    )
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if filepath is not None:
        fig.savefig(filepath)
    if show:
        plt.show()
    return fig, ax


def display_cv_scores(
    cv_scores, figsize=FIGSIZE, hline=True,
    title="", bottom=-1, top=1,
    filepath=None, show=False
):
    """"""
    if isinstance(cv_scores, dict):
        cv_scores = pd.DataFrame(cv_scores)
    # 
    fig, ax = plt.subplots(figsize=figsize)
    cv_unique = cv_scores["model"].unique()
    for sc in cv_unique:
        cv_score = cv_scores[cv_scores["model"] == sc]
        ax.plot(cv_score["score"], cv_score["mean"], label=f"model {sc}", marker="o")
    if hline:
        ax.axhline(0.5, color="g", linestyle="--")
        ax.axhline(0, color="b", linestyle="--")
    ax.set_ylim(bottom, top)
    ax.set_xticks([i for i in range(len(cv_score))])
    ax.set_xticklabels(
        cv_score["score"], rotation=45, ha='right', rotation_mode='anchor'
    )
    ax.set_title(title)
    ax.legend() if len(cv_unique) < 10 else None
    fig.tight_layout()
    if filepath is not None:
        fig.savefig(filepath)
    if show:
        plt.show()
    return fig, ax


def display_confusion_matrix(
    observed, predicted, labels=None, cmap="Blues",
    figsize=FIGSIZE, title="", filepath=None, show=False,
    add_cf_matrix=False
):
    """"""
    if labels is None:
        labels = [str(i) for i in np.unique(observed)]
    # Confusion matrix values
    cf_matrix = confusion_matrix(observed, predicted)
    cf_matrix_norm = confusion_matrix(observed, predicted, normalize="true")
    # To plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cf_matrix, interpolation='nearest', cmap=plt.get_cmap(cmap))
    # x & y labels
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks, labels, rotation=45)
    ax.set_yticks(tick_marks, labels)
    # Format
    thresh = cf_matrix.max() / 2
    for i, j in itertools.product(range(cf_matrix.shape[0]), range(cf_matrix.shape[1])):
        ax.text(j, i, f"{cf_matrix[i, j]:d}\n({cf_matrix_norm[i, j]*100:.0f}%)",
                horizontalalignment="center",
                color="white" if cf_matrix[i, j] > thresh else "black")

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.colorbar(im)
    fig.suptitle(title)
    fig.tight_layout()
    if filepath is not None:
        fig.savefig(filepath)
    if show:
        plt.show()
    if add_cf_matrix:
        return fig, ax, cf_matrix, cf_matrix_norm
    return fig, ax


def display_mean_confusion_matrix(
    obs_pred_list, labels=None, cmap="Blues",
    figsize=FIGSIZE, title="", filepath=None, show=False
):
    """"""
    if labels is None:
        labels = [str(i) for i in np.unique(obs_pred_list[0][0])]
    # List of confusion matrix
    cf_matrix_list = list()
    for i, (observed, predicted) in enumerate(obs_pred_list):
        cf_matrix_i = confusion_matrix(observed, predicted)
        cf_matrix_list.append(cf_matrix_i)
    # Mean & Std & norm confusion matrix
    n_cf_matrix = np.array(cf_matrix_list)
    cf_matrix_mean = n_cf_matrix.mean(axis=0)
    cf_matrix_std = n_cf_matrix.std(axis=0)
    row_sum = cf_matrix_mean.sum(axis=1).reshape(2, -1)  # true
    cf_matrix_norm = cf_matrix_mean / row_sum  # norm on true

    # To plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cf_matrix_mean, interpolation='nearest', cmap=plt.get_cmap(cmap))
    # x & y labels
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks, labels, rotation=45)
    ax.set_yticks(tick_marks, labels)
    # Format
    thresh = cf_matrix_mean.max() / 2
    for i, j in itertools.product(range(cf_matrix_mean.shape[0]), range(cf_matrix_mean.shape[1])):
        text = f"{cf_matrix_mean[i, j]:.0f}"u'\u00b1'f"{cf_matrix_std[i, j]:.1f}\n"
        text += f"({cf_matrix_norm[i, j]*100:.0f}%)"
        ax.text(j, i, text,
                horizontalalignment="center",
                color="white" if cf_matrix_mean[i, j] > thresh else "black")

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    fig.colorbar(im)
    fig.suptitle(title)
    fig.tight_layout()
    if filepath is not None:
        fig.savefig(filepath)
    if show:
        plt.show()

    return fig, ax


def display_raw_importance(
    raw_importance, figsize=FIGSIZE,
    title="Feature importances using MDI",
    ylabel="Decrease in accuracy score",
    violin=False, filepath=None, show=False
):
    """raw_importance: pandas.DataFrame, or dict"""
    if isinstance(raw_importance, dict):
        raw_importance = pd.DataFrame(raw_importance)
    #
    fig, ax = plt.subplots(figsize=figsize)
    x_ticks = [i for i in range(len(raw_importance.columns))]
    if not violin:
        ax.boxplot(raw_importance, labels=raw_importance.columns)
    else:
        ax.violinplot(raw_importance, positions=x_ticks, showmeans=False, showmedians=True)
    ax.set_xticks([i+1 for i in range(len(raw_importance.columns))])
    ax.set_xticklabels(
        raw_importance.columns, rotation=45,
        ha='right', rotation_mode='anchor'
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if filepath is not None:
        fig.savefig(filepath)
    if show:
        plt.show()

    return fig, ax

    
def display_mdi_importance(
    mdi_importance, figsize=FIGSIZE,
    title="Feature importances using MDI",
    ylabel="Mean decrease in impurity",
    filepath=None, show=False
):
    """mdi_importance: pandas.DataFrame, or dict"""
    if isinstance(mdi_importance, dict):
        mdi_importance = pd.DataFrame(mdi_importance)
    #
    mean_values = mdi_importance["importances_mean"]
    std_values = mdi_importance["importances_std"]
    x_values = mdi_importance["colnames"]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x_values, mean_values, yerr=std_values, align="center", ecolor="black")
    ax.set_xticks([i for i in range(len(x_values))])
    ax.set_xticklabels(
        x_values, rotation=45,
        ha='right', rotation_mode='anchor'
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if filepath is not None:
        fig.savefig(filepath)
    if show:
        plt.show()

    return fig, ax


def display_permutation_importance(
    permutation_importance, figsize=FIGSIZE,
    title="Feature importances using permutation",
    ylabel="Mean accuracy decrease",
    filepath=None, show=False
):
    """"""
    fig, ax = display_mdi_importance(
        mdi_importance=permutation_importance,
        figsize=figsize, title=title, ylabel=ylabel,
        filepath=filepath, show=show
    )
    return fig, ax


def display_boruta_importance(
    boruta_importance, treshold, n_trials,
    figsize=FIGSIZE, title="", ylabel="Number of Hit",
    lower_color="r", upper_color="g",
    filepath=None, show=False
):
    """"""
    #
    lower_treshold = treshold
    upper_treshold = n_trials - treshold
    count_values = list(boruta_importance.values())
    x_values = list(boruta_importance.keys())
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x_values, count_values, align="center")
    ax.set_xticks([i for i in range(len(x_values))])
    ax.set_xticklabels(
        x_values, rotation=45,
        ha='right', rotation_mode='anchor'
    )
    ax.axhline(upper_treshold, color=upper_color, linestyle="--")
    ax.axhline(lower_treshold, color=lower_color, linestyle="--")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    if filepath is not None:
        fig.savefig(filepath)
    if show:
        plt.show()

    return fig, ax


"""
def display_rf_summary_shap(
    estimator, x, feature_names=None,
    figsize=FIGSIZE, autosize=True,
    filepath=None, show=False,
):
    """"""
    fig, ax = plt.subplots(figsize=figsize)
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(x)
    shap.summary_plot(
        shap_values, x, 
        auto_size_plot=autosize, show=False
    )
    #plt.tight_layout()
    if filepath is not None:
        plt.savefig(filepath)
    if show:
        plt.show()

    return fig, ax
"""


if __name__ == "__main__":
    pass
