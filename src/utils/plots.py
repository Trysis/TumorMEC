#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This scripts contains plot function and plot saving functions."""

import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.metrics import r2_score

# Decision Tree graph
import collections
import pydotplus

# Local modules
import auxiliary

__author__ = "Roude JEAN MARIE"
__email__ = "roude.bioinfo@gmail.com"


def node_color(
    dot_data, colors=('#33aaff', '#ff664d'),
    by="entropy", attr="label", n_labels=2
):
    """Apply the selected colors on a decision tree
    with transparency depending on node impurity.

    -----
    Exemple:
    
        dtree = DecisionTreeClassifier(...)  # object
        columns = ...  # features in tuple(str, ...)
        labels = ... # label class in tuple(str, ...)
    
        dot_data = tree.export_graphviz(
            dtree,
            feature_names=columns,
            class_names=labels,
        )
        dot_data = node_color(dot_data, by="entropy")
        graph = graphviz.Source(dot_data, format="png")
        graph.render(...)  # save decision tree with new color

    -----
    dot_data: str
        decision tree dot data representation
        from sklearn.tree.export_graphviz

    colors: tuple -> tuple(str, str, ...)
        Applied color for each child from a node
        in the corresponding order, such that a tree
        predicting two classes needs two colors.

    by: str
        criterion for node impurity such as
        gini, entropy. It will be used
        to apply transparency such that a complete disorder
        will be completely transparent and no disorder
        won't be at all transparent.

    attr: str
        node label to access and retrieve data to
        perform modifications on.

    Returns: str
        dot data with modified color

    """
    criterion_str = f"{by} = "
    sep = "\\n"

    # Graph from dot data
    graph_dot = pydotplus.graphviz.graph_from_dot_data(dot_data)
    edges = collections.defaultdict(list)

    for edge in graph_dot.get_edge_list():
        edges[edge.get_source()].append(int(edge.get_destination()))

    for edge in edges:
        edges[edge].sort()
        for i in range(n_labels):
            node = graph_dot.get_node(str(edges[edge][i]))[0]
            node_label = node.get_attributes()[attr]
            entropy = float(node_label.split(criterion_str)[1].split(sep)[0])
            transparency = hex(int((1 - entropy) * 255))[2:]
            if len(transparency) == 1:
                transparency = '1' + transparency
            node.set_fillcolor(colors[i] + transparency)

    # Final decision tree graph
    return graph_dot.to_string()


def to_pdf(filepath, figures, then_close=False, bbox_inches=None):
    """Save a list of figure in pdf format

    filepath: str
        Path indicating where to create the file

    figures: Generator or List -> List(matplotlib.figure.Figure)
        A list of matplotlib figures

    Exemple:
        fig1 = plt.figure(figsize=(7, 8))  # plot 1
        plt.plot([1, 2, 3], [2, 6, 10]
        fig2 = plt.figure(figsize=(7, 8))  # plot 2
        plt.scatter([1, 2, 3, 4], [1, 2, 5, 7])
        figures = [plt.figure(n) for n in plt.get_fignums()]
        # Save plots to pdf
        to_pdf("test.pdf", figures)

    """
    filepath = auxiliary.replace_extension(filepath, "pdf")
    with PdfPages(filepath) as pdf:
        # Each figure is saved in pdf object
        for fig in figures:
            fig.savefig(pdf, format='pdf', bbox_inches=bbox_inches)
            if then_close:
                plt.close(fig)


def legend_patch(label, color="none"):
    """Returns the corresponding label with a Patch object
    for matplotlib legend purposes.
    
    label: str
        Character chain specifying the label
    color: str
        Corresponding color attribute for label

    Returns: tuple -> tuple(str, matplotlib.patches.Patch)
        Specified label with Patch

    """
    return label, mpatches.Patch(color=color, label=label)


def single_plot(
    yvalues, xvalues=None, scale=None, mode="plot",
    scale_x="linear", scale_y="linear", title="",
    xlabel="", ylabel="", label="", alphas=(1,),
    xleft=None, xright=None, ytop=None, ybottom=None, normalize=False,
    figure=None, figsize=None,  anchor_x=1.05, anchor_y=0.9,
    color=None, r2=None, showY=True, showR2=True,
    save_to=None, filename="plot", ext="png",
    forcename=False, overwrite=True,
    ignore_nan=True, **kwargs
):
    """Generate a specified figure from a set of values.

    yvalues: Array-like (numpy.ndarray)
        Values to plot, correspond to y axis

    xvalues: Array-like (numpy.ndarray)
        Values in x axis

    scale: str
        Chosen scale for observed and predicted values
        "linear", "log", ...

    mode: str
        Selected mode for plot taking values such as 
        "plot" -> line plot
        "bar" -> barplot plot
        "hist" -> histogram plot
        "scatter" -> scatter plot
        "violin" -> violin plot

    scale_x_, scale_y: str, default="linear"
        Chosen scale on x and y axis.

    title, xlabel, ylabel: str
        Title, x-axis and y-axis labels to assign to
        the figure

    label: str
        Associated label to the plot (for legend purpose).

    alphas: tuple(float), optional
        alpha values for the plot

    xleft, xright: float, optional
        lower and upper x limit

    ybottom, ytop: int, optional
        lower and upper y limit

    normalize: bool
        Normalize yvalues so that it is bound to [0; 1]

    figure: matplotlib.figure.Figure -> (fig, ax)
        A tuple corresponding to a Figure returned by
        matplotlib.figure.figure, the axis {ax} should
        correspond to only one

    figsize: tuple -> tuple(float, float), default=(8, 7)
        Width and height in inches.

    anchor_x, anchor_y: float, float
        Box legend position visible with {showY} and {showR2}

    color: tuple, default=blue (#1f77b4)
        Associated color for each {values} plots

    r2: float
        R2 metric. if not specified, a pearson correlation
        will be calculated between {xvalues} and {yvalues}.

    showY: bool
        show legend parameters associated with the metric

    showR2: bool
        show R2 values for each {values}

    save_to: str
        Directory to save plot to

    filename: str
        Name of the file to create

    ext: str
        Extension of the file to create

    forcename: bool
        If True, then filename will exactly be the same
        as specified and only the extension will be appended.
        Else, extension is replaced by detecting the last "."
        character to replace it.

    overwrite: str
        If a plot have the same name as our, should we
        overwrite ? If not, it creates a filename with
        an appended suffix

    ignore_nan: bool
        Should we ignore nan value ? When plotting,
        and calculating metrics such as mean, median
        and std

    Returns: tuple (matplotlib.figure.Figure, matplotlib.axes.Axes)
        Figure and Axes for graphical purposes

    """
    # Figure size
    figsize = (8, 7) if figsize is None else figsize

    # Plot mode selection
    plotting_mode = {
        "plot": lambda ax: ax.plot,
        "bar": lambda ax: ax.bar,
        "hist": lambda ax: ax.hist,
        "scatter": lambda ax: ax.scatter,
        "violin": lambda ax: ax.violinplot
    }

    # Check mode
    if mode not in plotting_mode.keys():
        raise Exception(f"Selected mode={mode} is invalid")

    # x-axis equals indices if not set
    if xvalues is None:
        xvalues = np.arange(len(yvalues))

    alphas = (1,) if alphas is None else alphas
    alphas = (alphas, ) if isinstance(alphas, (int, float)) else tuple(alphas)

    # Optional arguments
    if mode.startswith("hist"):
        kwargs["bins"] = kwargs.get("bins", 100)
    kwargs["color"] = "#1f77b4" if color is None else color

    xticks = kwargs.pop("xticks") if kwargs.get("xticks", None) is not None else None
    yticks = kwargs.pop("yticks") if kwargs.get("yticks", None) is not None else None
    grid = kwargs.pop("grid") if kwargs.get("grid", False) else False

    # Conversion
    if not isinstance(yvalues, np.ndarray):
        yvalues = np.array(yvalues)

    if not isinstance(xvalues, np.ndarray):
        xvalues = np.array(xvalues)

    # Metrics
    isnotnan = ~np.isnan(yvalues)  # index of not NaN values
    if normalize:  # Normalization of value, if specified
        yvalues = auxiliary.min_max_normalization(
            yvalues, 0, 1, ignore_nan=ignore_nan
        )

    # Observed and Predicted : Mean, std, median
    mean_yvalues = yvalues.mean() if not ignore_nan else np.nanmean(yvalues)
    std_yvalues = yvalues.std() if not ignore_nan else np.nanstd(yvalues)
    median_yvalues = np.median(yvalues) if not ignore_nan else np.nanmedian(yvalues)

    # Plot depending on mode
    fig, ax = plt.subplots(figsize=figsize) if figure is None else figure
    if (mode == "plot"):
        plotting_mode[mode](ax)(xvalues, yvalues, label=label, alpha=alphas[0], **kwargs)
    # Histogram
    elif (mode == "hist"):
        bins=kwargs["bins"]
        plotting_mode[mode](ax)(yvalues, bins=bins, label=label, alpha=alphas[0])
    # Barplot
    elif (mode == "bar"):
        plotting_mode[mode](ax)(xvalues, yvalues, label=label, alpha=alphas[0], **kwargs)
    # Violin (TODO: More option for violin plot)
    elif (mode == "violin"):
        v1 = plotting_mode[mode](ax)(yvalues, positions=[0], **kwargs)
        v1_label = [mpatches.Patch(color=v1["bodies"][0].get_facecolor().flatten()), label]
        main_legend = ax.legend(*zip(*v1_label), loc=2)
        ax.add_artist(main_legend)
    # Scatter plot
    elif (mode == "scatter"):
        plotting_mode[mode](ax)(xvalues, yvalues, alpha=alphas[0], **kwargs)

    # Main Legend
    if True:
        handles, labels = ax.get_legend_handles_labels()
        if (handles != []) & (labels != []):
            main_legend = ax.legend(handles, labels, loc="best")
            ax.add_artist(main_legend)

        # Show statistics about {yvalues}
        if showY:
            handles_y, labels_y = [], []
            mean_yvalues_label, mean_yvalues_patch = legend_patch(f"mean = {mean_yvalues:.3f}")
            std_yvalues_label, std_yvalues_patch = legend_patch(f"std = {std_yvalues:.3f}")
            median_yvalues_label, median_yvalues_patch = legend_patch(f"median = {median_yvalues:.3f}")

            handles_y.extend([mean_yvalues_patch, std_yvalues_patch, median_yvalues_patch])
            labels_y.extend([mean_yvalues_label, std_yvalues_label, median_yvalues_label])

            msm_observed_legend = ax.legend(
                handles_y, labels_y, title=label, loc="center",
                handlelength=0, handletextpad=0, borderaxespad=0,
                bbox_to_anchor=(anchor_x, anchor_y)
            )

            ax.add_artist(msm_observed_legend)  # Add legend to artist

        # Show correlation value between x and y values
        if showR2:  # R2
            r2 = r2 if r2 else r2_score(xvalues[isnotnan], yvalues[isnotnan])
            handles_r2, labels_r2 = [], []
            r2_label, r2_patch = legend_patch(f"R2 = {r2:.4f}")
            handles_r2.append(r2_patch)
            labels_r2.append(r2_label)

            r2_legend = ax.legend(
                handles_r2, labels_r2, loc="upper right",
                handlelength=0, handletextpad=0
            )

            # Add legend
            ax.add_artist(r2_legend)

    # Set scale selected by user
    if scale is not None:
        scale_x = scale
        scale_y = scale

    ax.set_xscale(scale_x)
    ax.set_yscale(scale_y)

    # Option for specific mode
    if (mode == "scatter"):
        # Range to have xlim=ylim
        xy_lim = auxiliary.min_max(ax.get_xlim() + ax.get_ylim())
        ax.set_xlim(xy_lim)
        ax.set_ylim(xy_lim)

    # Set figure label, limit and legend
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    # Label
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Limit in x and y
    ax.set_xlim(left=xleft, right=xright)
    ax.set_ylim(top=ytop, bottom=ybottom)
    # Grid
    ax.grid(grid)

    if save_to is not None:
        if not auxiliary.isdir(save_to):
            raise Exception("Specified directory does not exists")

        ext = ext.replace(".", "")
        save_to = auxiliary.to_dirpath(save_to)
        filepath = save_to + filename + "." + ext
        if not forcename:
            root, _ = os.path.splitext(filename)
            # Save file to
            filename = auxiliary.replace_extension(root, ext)
            filepath = save_to + filename if overwrite else \
                       auxiliary.filepath_with_suffix(save_to + filename)

        plt.savefig(filepath, bbox_inches = 'tight')

    return fig, ax


def dual_plot(
    observed, predicted, indices=None, scale="linear", mode="plot",
    title="", xlabel="", ylabel="", metric="",
    label_1="observed", label_2="predicted", alphas=(1, 1),
    xleft=None, xright=None, ytop=None, ybottom=None, normalize=False,
    figure=None, figsize=None, anchor_x=1.06, anchor_y=0.88,
    color_1=None, color_2=None, r2=None, loss=None,
    showR2=True, showLoss=True, showObs=True, showPred=True, showDelta=True,
    save_to=None, filename="plot", ext="png", forcename=False, overwrite=True,
    ignore_nan=True, **kwargs
):
    """Generate a specified figure.

    observed: Iterator (numpy.ndarray)
        Observed values

    predicted: Iterator (numpy.ndarray)
        Predicted values

    indices: Iterator (numpy.ndarray)
        Observed and predicted indices to perform plots on
        values need to be in array observed/predicted indices range.

    scale: str
        Chosen scale for observed and predicted values
        "linear", "log", ...

    mode: str
        Selected mode for plot taking values such as 
        "plot" -> line plot
        "bar" -> barplot plot
        "hist" -> histogram plot
        "hist2d" -> histogram2d plot
        "scatter" -> scatter plot
        "violin" -> violin plot

    title, xlabel, ylabel: str
        Title, x-axis and y-axis labels to assign to the figure

    metric: str
        Name of the supervised metric

    label_1, label_2: str, str
        Label associated with {observed} and {predicted} values

    alphas: tuple -> tuple(float), optional
        alpha values for the {observed} and {predicted} values

    xleft, xright: float, optional
        lower and upper x limit

    ybottom, ytop: int, optional
        lower and upper y limit

    normalize: bool
        Normalize values so that it is bound to [0; 1] values
        or [-1; 1] values if {delta}=True

    figure: matplotlib.figure.Figure -> (fig, ax)
        A tuple corresponding to a Figure returned by
        matplotlib.figure.figure, the axis {ax} should
        correspond to one matplotlib.axes.Axes

    figsize: tuple -> tuple(float, float), optional
        Width and height in inches.

    anchor_x, anchor_y: float, default=1.06; 0.88
        Box legend position visible with {showY} and {showR2}

    color_1, color_2: matplotlib color, matplotlib color
        color associated to {observed} and {predicted}

    r2, loss: float, float
        R2 and Loss metric, if not specified it
        Pearson correlation and MSE will be calculated
        betweend observed and predictions

    showR2, showLoss, showObs, showPred, showDelta: bool
        Should the associated metric or label legend be showed ?

    save_to: str
        Directory to save plot to

    filename: str
        Name of the file to create

    ext: str
        Extension of the file to create

    forcename: bool
        It True, then filename will exactly be the same
        as specified. Only the extension will be appended.
        Else, extension is replaced by detecting the last "."
        character to replace it.

    overwrite: str
        If a plot have the same name as our, should we
        overwrite ? If not, it creates a filename with
        an appended suffix

    ignore_nan: bool
        If observed or predicted values contains NaN,
        should we ignore them ?

    Returns: tuple (matplotlib.figure, matplotlib.axes.Axes)
        Figure and Axes for graphical purposes

    """
    # Plot mode selection
    plotting_mode = {
        "plot": lambda ax: ax.plot,
        **dict.fromkeys(["bar", "delta_bar"], lambda ax: ax.bar),
        "hist": lambda ax: ax.hist,
        "scatter": lambda ax: ax.scatter,
        "hist2d": lambda ax: ax.hist2d,
        "violin": lambda ax: ax.violinplot
    }
    # Check mode
    if mode not in plotting_mode.keys():
        raise Exception("Selected mode is invalid")

    # Set indices if not already set
    if indices is None:
        indices = np.arange(len(observed))

    alphas = (1,) if alphas is None else alphas
    alphas = (alphas, ) if isinstance(alphas, (int, float)) else tuple(alphas)
    alphas = alphas + (1,)
    delta = True if mode.startswith("delta") else False

    # Optional arguments
    if mode.startswith("hist"):
        kwargs["bins"] = kwargs.get("bins", 100)  # if mode == "hist"/"hist2d"
        kwargs["cmap"] = kwargs.get("cmLoadap", plt.cm.jet)
        kwargs["norm"] = kwargs.get("norm", mcolors.LogNorm())

    xticks = kwargs.pop("xticks") if kwargs.get("xticks", None) is not None else None
    yticks = kwargs.pop("yticks") if kwargs.get("yticks", None) is not None else None
    grid = kwargs.pop("grid") if kwargs.get("grid", False) else False

    if len(observed) != len(predicted):
        raise Exception("Observed and Predicted value should have the same length")

    # Conversion
    if not isinstance(indices, np.ndarray):
        indices = np.array(indices)

    if not isinstance(observed, np.ndarray):
        observed = np.array(observed)

    if not isinstance(predicted, np.ndarray):
        predicted = np.array(predicted)

    # Delta (Observed - Predicted)
    delta_values = (observed - predicted).copy()

    # Metrics
    isnotnan = ~np.isnan(observed) & ~np.isnan(predicted)
    r2 = r2 if r2 else r2_score(observed[isnotnan], predicted[isnotnan])
    loss = loss if loss else ((observed[isnotnan] - predicted[isnotnan]) ** 2).mean()

    # Normalization needed before mean, std, median calculation
    if (normalize):
        observed = auxiliary.normalization_min_max(observed, 0, 1, ignore_nan=ignore_nan)
        predicted = auxiliary.normalization_min_max(predicted, 0, 1, ignore_nan=ignore_nan)
        delta_values = auxiliary.normalization_min_max(delta_values, -1, 1, ignore_nan=ignore_nan) 

    # Observed and Predicted : Mean, std, median
    metrics_observed = auxiliary.get_metrics(observed, ignore_nan=ignore_nan)
    mean_observed, std_observed, median_observed = \
        metrics_observed["mean"], metrics_observed["std"], metrics_observed["median"]

    metrics_predicted = auxiliary.get_metrics(predicted, ignore_nan=ignore_nan)
    mean_predicted, std_predicted, median_predicted = \
        metrics_predicted["mean"], metrics_predicted["std"], metrics_predicted["median"]

    # Delta (Observed - Predicted) : Mean, std, median
    metrics_delta = auxiliary.get_metrics(delta_values, ignore_nan=ignore_nan)
    mean_delta, std_delta, median_delta = \
        metrics_delta["mean"], metrics_delta["std"], metrics_delta["median"]

    # Plot depending on mode
    fig, ax = plt.subplots(figsize=figsize) if figure is None else figure
    label_observed, label_predicted = label_1, label_2
    if (mode == "plot"):
        plotting_mode[mode](ax)(indices, observed[indices], label=label_observed, color=color_1, alpha=alphas[0], **kwargs)
        plotting_mode[mode](ax)(indices, predicted[indices], label=label_predicted,  color=color_2, alpha=alphas[1], **kwargs)
    elif (mode == "hist"):
        bins=kwargs["bins"]
        plotting_mode[mode](ax)(observed[indices], bins=bins, label=label_observed, color=color_1, alpha=alphas[0])
        plotting_mode[mode](ax)(predicted[indices], bins=bins, label=label_predicted, color=color_2,  alpha=alphas[1])
    elif (mode == "bar"):
        plotting_mode[mode](ax)(indices, observed[indices], label=label_observed, color=color_1, alpha=alphas[0], **kwargs)
        plotting_mode[mode](ax)(indices, predicted[indices], label=label_predicted, color=color_2, alpha=alphas[1], **kwargs)
    elif (mode == "violin"):
        v1 = plotting_mode[mode](ax)(observed[indices], positions=[0], **kwargs)
        v2 = plotting_mode[mode](ax)(predicted[indices], positions=[0.5], **kwargs)
        labels = []
        labels.append((mpatches.Patch(color=v1["bodies"][0].get_facecolor().flatten()), label_observed))
        labels.append((mpatches.Patch(color=v2["bodies"][0].get_facecolor().flatten()), label_predicted))
        main_legend = ax.legend(*zip(*labels), loc=2)
        ax.add_artist(main_legend)
    elif (mode == "scatter"):
        plotting_mode[mode](ax)(observed[indices], predicted[indices], color=color_1, alpha=alphas[0], **kwargs)
    elif (mode == "hist2d"):
        plotting_mode[mode](ax)(observed[indices], predicted[indices], **kwargs)
    elif (mode == "delta_bar"):
        delta_values_idx = delta_values[indices]
        plotting_mode[mode](ax)(indices, delta_values_idx, color=color_1, alpha=alphas[0], **kwargs)

    # Main Legend
    handles, labels = ax.get_legend_handles_labels()
    if (handles != []) & (labels != []):
        main_legend = ax.legend(handles, labels, loc="upper left")
        ax.add_artist(main_legend)

    # R2 & Loss Legend
    if showR2 or showLoss:
        handles_r2loss, labels_r2loss = [], []
        if showR2:  # R2
            r2_label, r2_patch = legend_patch(f"R2 = {r2:.4f}")
            handles_r2loss.append(r2_patch)
            labels_r2loss.append(r2_label)
        if showLoss:  # Loss
            loss_label, loss_patch = legend_patch(f"loss = {loss:.4f}")
            handles_r2loss.append(loss_patch)
            labels_r2loss.append(loss_label)

            r2_loss_legend = ax.legend(
                handles_r2loss, labels_r2loss, loc="upper right",
                handlelength=0, handletextpad=0
            )

        # Add legend
        ax.add_artist(r2_loss_legend)

    # Mean, std, median Legend
    if delta:
        if showDelta:
            # Delta : (Observed - Predicted)
            handles_delta, labels_delta = [], []
            mean_delta_label, mean_delta_patch = legend_patch(f"mean = {mean_delta:.3f}")
            std_delta_label, std_delta_patch = legend_patch(f"std = {std_delta:.3f}")
            median_delta_label, median_delta_patch = legend_patch(f"median = {median_delta:.3f}")

            handles_delta.extend([mean_delta_patch, std_delta_patch, median_delta_patch])
            labels_delta.extend([mean_delta_label, std_delta_label, median_delta_label])

            msm_delta_legend = fig.legend(
                handles_delta, labels_delta, title=f"Delta{metric}",
                handlelength=0, handletextpad=0, borderaxespad=0,
                bbox_to_anchor=(anchor_x, anchor_y)
            )

            # Add legend
            ax.add_artist(msm_delta_legend)
    else:
        if showObs:  # Observed
            handles_obs, labels_obs = [], []
            mean_observed_label, mean_observed_patch = legend_patch(f"mean = {mean_observed:.3f}")
            std_observed_label, std_observed_patch = legend_patch(f"std = {std_observed:.3f}")
            median_observed_label, median_observed_patch = legend_patch(f"median = {median_observed:.3f}")

            handles_obs.extend([mean_observed_patch, std_observed_patch, median_observed_patch])
            labels_obs.extend([mean_observed_label, std_observed_label, median_observed_label])

            msm_observed_legend = fig.legend(
                handles_obs, labels_obs, title="observed",
                handlelength=0, handletextpad=0, borderaxespad=0,
                bbox_to_anchor=(anchor_x-0.7, anchor_y)
            )

            # Add legend
            ax.add_artist(msm_observed_legend)

        if showPred:  # Predicted
            handles_pred, labels_pred = [], []
            mean_predicted_label, mean_predicted_patch = legend_patch(f"mean = {mean_predicted:.3f}")
            std_predicted_label, std_predicted_patch = legend_patch(f"std = {std_predicted:.3f}")
            median_predicted_label, median_predicted_patch = legend_patch(f"median = {median_predicted:.3f}")

            handles_pred.extend([mean_predicted_patch, std_predicted_patch, median_predicted_patch])
            labels_pred.extend([mean_predicted_label, std_predicted_label, median_predicted_label])

            msm_predicted_legend = fig.legend(
                handles_pred, labels_pred, title="predicted",
                handlelength=0, handletextpad=0, borderaxespad=0,
                bbox_to_anchor=(1.06, 0.7)
            )

            # Add legend
            ax.add_artist(msm_predicted_legend)

    # Scale selected by user
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    # Option for specific mode
    if (mode == "scatter"):
        # Range to have xlim=ylim
        xy_lim = auxiliary.min_max(ax.get_xlim() + ax.get_ylim())
        ax.set_xlim(xy_lim)
        ax.set_ylim(xy_lim)
    elif (mode == "delta_bar"):
        # Horizontal separation line
        ax.axhline(y=0, color='orange', linestyle = '--')
        # Range to have ylim(a, b) with a=max(absolute(y)) and a=b
        highest_value = max(np.abs(ax.get_ylim()))
        ax.set_ylim((-highest_value, highest_value))

    # Set figure label, limit and legend
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    # Label
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Limit in x and y
    ax.set_xlim(left=xleft, right=xright)
    ax.set_ylim(top=ytop, bottom=ybottom)
    # Grid
    ax.grid(grid)

    if save_to is not None:
        if not auxiliary.isdir(save_to):
            raise Exception("Specified directory does not exists")

        ext = ext.replace(".", "")
        save_to = auxiliary.to_dirpath(save_to)
        filepath = save_to + filename + "." + ext
        if not forcename:
            root, _ = os.path.splitext(filename)
            # Save file to
            filename = auxiliary.replace_extension(root, ext)
            filepath = save_to + filename if overwrite else \
                    auxiliary.filepath_with_suffix(save_to + filename)

        plt.savefig(filepath, bbox_inches = 'tight')

    return fig, ax


def plot(
    *values, indices=None, scale=None, mode="plot",
    scale_x="linear", scale_y="linear",
    title="", xlabel="", ylabel="", label=("",), alphas=(1,),
    xleft=None, xright=None, ytop=None, ybottom=None, normalize=False,
    figure=None, figsize=None, anchor_x=1.05, anchor_y=0.9,
    loc_label="upper left", loc_r2="upper right",
    color=None, color_palette="Set1", showY=True, showR2=True,
    save_to=None, filename="plot", ext="png", forcename=False, overwrite=True,
    ignore_nan=True, **kwargs
):
    """Generate a specified figure from a set of values.

    values: list -> list(Array-like)
        Values to plot, correspond to y axis

    indices: list -> list(Array-like)
        Values in x axis

    scale: str
        Chosen {scale} for observed and predicted values
        applied on x and y axis.
        {scale} can take value such as "linear", "log", etc
        see matplotlib documentation.

    mode: str, default="plot"
        Selected mode for plot taking values such as 
        "plot" -> line plot
        "bar" -> barplot plot
        "hist" -> histogram plot
        "scatter" -> scatter plot
        "violin" -> violin plot

    scale_x_, scale_y:
        Chosen scale on x and y axis.

    title, xlabel, ylabel: str
        Title, x-axis and y-axis labels to assign to
        the figure

    label: list -> list(str)
        Associated label to the plot (for legend purpose).

    alphas: tuple(float), optional
        alpha values for the plot

    xleft, xright: float, optional
        lower and upper x limit

    ytop, ybottom: int, optional
        lower and upper y limit

    normalize: bool
        Normalize values so that it is bound to [0; 1]

    figure: matplotlib.figure.Figure -> (fig, ax)
        A tuple corresponding to a Figure returned by
        matplotlib.figure.figure, the axis {ax} should
        correspond to one matplotlib.axes.Axes

    figsize: tuple -> tuple(float, float), optional
        Width and height in inches.

    anchor_x, anchor_y: float, default=1.05; 0.9
        Box legend position visible with {showY} and {showR2}

    loc_label, loc_r2: str, default="upper left"; "upper right"
        Matplotlib legend location for when {showY=True}
        and {showR2=True}

    color: tuple, optional
        Associated color for each {values} plots

    color_palette: matplotlib.colors.ListedColormap
        Associated palette to the plot. Is used when
        each {values} does not have an attributed color.

    showY: bool
        show legend parameters associated with the metric

    showR2: bool
        show R2 values for each {values}

    save_to: str, default=None
        Directory to save plot to. If None, no saving
        is applied.

    filename: str
        Name of the file to create

    ext: str
        Extension of the file to create

    forcename: bool
        If True, then filename will exactly be the same
        as specified and only the extension will be appended.
        Else, extension is replaced by detecting the last "."
        character to replace it.

    overwrite: str
        If a plot have the same name as our, should we
        overwrite ? If not, it creates a filename with
        an appended suffix

    ignore_nan: bool
        Should we ignore nan value ? When plotting,
        and calculating metrics such as mean, median
        and std    

    Returns: tuple (matplotlib.figure, matplotlib.axes.Axes)
        Figure and Axes for graphical purposes

    """
    # Number of observations to plot
    n_obs = len(values)
    # Color palette
    palette = sns.color_palette(color_palette)
    # Figure size
    figsize = (8, 7) if figsize is None else figsize

    # Plot mode selection
    plotting_mode = {
        "plot": lambda ax: ax.plot,
        "bar": lambda ax: ax.bar,
        "hist": lambda ax: ax.hist,
        "scatter": lambda ax: ax.scatter,
        "violin": lambda ax: ax.violinplot
    }

    # Check mode
    if mode not in plotting_mode.keys():
        raise Exception("Selected mode is invalid")

    if n_obs == 0:
        raise Exception(
            "At least one argument should be provided for {values}"
        )

    # Check if each {*values} provided is of instance np.ndarray
    # converts them if possible.
    values_length = []
    values_list = list(values)
    for idx, val in enumerate(values):
        if not isinstance(val, np.ndarray):
            try:
                val_np = np.array(val)
                values_list[idx] = val_np if not normalize else \
                                   auxiliary.min_max_normalization(
                                       val_np, 0, 1, ignore_nan=ignore_nan
                                   )
            except:
                raise Exception(
                    "Array-like values should be "
                    "provided for {values} argument"
                )
        values_length.append(len(val))

    values = tuple(values_list)

    # x-axis equals indices if not set
    if indices is None:
        indices = [np.arange(length) for length in values_length]
    elif all(isinstance(ind, (int, float)) for ind in indices):
        indices = [indices]
    for idx, ind in enumerate(indices):
        if not isinstance(ind, np.ndarray):
            try:
                indices[idx] = np.array(ind)
            except:
                raise Exception(
                    "Could not convert data to "
                    "numpy.ndarray, check {indices} "
                    "argument."
                )

    if (len(indices) == 1) & (n_obs > len(indices)):
        if len(indices[0]) < max(values_length):
            raise Exception("indices should be set for each ticks position")

        indices_per_val = []
        for idx in range(n_obs):
            indices_per_val.append(indices[0][:len(values[idx])])
        indices = indices_per_val

    # alpha value for each {*values}
    alphas = (1,) if alphas is None else alphas
    alphas = (alphas, ) if isinstance(alphas, (int, float)) else tuple(alphas)
    alphas = alphas + (1,) * (n_obs-len(alphas)) if len(alphas) < n_obs else alphas

    # color value for each {*values}
    color = ("#1f77b4",) if color is None else color
    if len(color) < n_obs:
        color_tmp = []
        for idx in range(n_obs - len(color)):
            color_tmp.append(palette[idx%len(palette)])
        color = color + tuple(color_tmp)

    label = ("", ) if label is None else label
    label = (label, ) if isinstance(label, (str)) else tuple(label)
    label = label + ("",) * (n_obs-len(label)) if len(label) < n_obs else label

    ## Optional arguments
    # Bins
    if mode.startswith("hist"):
        kwargs["bins"] = kwargs.get("bins", (100,))
        if isinstance(kwargs["bins"], (float, int)):
            kwargs["bins_list"] = (kwargs["bins"],)
        else:
            kwargs["bins_list"] = tuple(kwargs["bins"])

        if len(kwargs["bins_list"]) < n_obs:
            kwargs["bins_list"] = kwargs["bins_list"] + (100, ) * (n_obs - len(kwargs["bins_list"]))

    # kwargs
    dodge = kwargs.pop("dodge") if "dodge" in kwargs else False
    spacing = kwargs.pop("spacing") if "spacing" in kwargs else 0
    if mode == "bar":
        kwargs["width"] = kwargs.get("width", 0.8)
        impair_n_obs = n_obs % 2 == 1
        bar_positions = (
            np.linspace(-n_obs / 2, n_obs / 2, num=n_obs) if impair_n_obs else
            np.setdiff1d(np.linspace(-n_obs / 2, n_obs / 2, num=n_obs+1), 0)
        )

        spacing_pos = []
        for idx, i in enumerate(bar_positions):
            if i < 0: spacing_pos.append(-spacing * np.abs(bar_positions[idx]))
            elif i > 0: spacing_pos.append(spacing * np.abs(bar_positions[idx]))
            else: spacing_pos.append(0)
        spacing_pos = np.array(spacing_pos)

    xticks = kwargs.pop("xticks") if kwargs.get("xticks", None) is not None else None
    yticks = kwargs.pop("yticks") if kwargs.get("yticks", None) is not None else None
    grid = kwargs.pop("grid") if kwargs.get("grid", False) else False

    # Plot depending on mode
    fig, ax = plt.subplots(figsize=figsize) if figure is None else figure
    _vl_label_list = []        
    for idx, val in enumerate(values):
        if (mode == "plot"):
            plotting_mode[mode](ax)(indices[idx], val, label=label[idx], alpha=alphas[idx], color=color[idx], **kwargs)
        elif (mode == "hist"):
            plotting_mode[mode](ax)(val, bins=kwargs["bins_list"][idx], label=label[idx], color=color[idx], alpha=alphas[idx])
        elif (mode == "bar"):
            if dodge:  # When there is multiple bar, shift them
                s = (kwargs["width"]) * (n_obs) * indices[idx]
                if n_obs % 2 == 0:
                    v = kwargs["width"]/2 if bar_positions[idx] < 0 else -kwargs["width"]/2
                    plotting_mode[mode](ax)(
                        indices[idx] + (indices[idx] * (spacing*n_obs + n_obs/2)) + int(bar_positions[idx]) * (kwargs["width"]) + spacing_pos[idx] + s + v,
                        val, label=label[idx], alpha=alphas[idx],
                        color=color[idx], **kwargs
                    )
                else:
                    plotting_mode[mode](ax)(
                        indices[idx] + (indices[idx] * (spacing*n_obs + n_obs/2)) + int(bar_positions[idx]) * (kwargs["width"]) + spacing_pos[idx] + s,
                        val, label=label[idx], alpha=alphas[idx],
                        color=color[idx], **kwargs
                    )
            else:
                plotting_mode[mode](ax)(indices[idx], val, label=label[idx], alpha=alphas[idx], color=color[idx], **kwargs)
        elif (mode == "violin"):
            vl = plotting_mode[mode](ax)(val, positions=[idx], **kwargs)
            vl_label = [mpatches.Patch(color=vl["bodies"][0].get_facecolor().flatten()), label]
            _vl_label_list.append(vl_label)
            if idx == n_obs - 1:
                main_legend = ax.legend(*zip(*_vl_label_list), loc=2)
                ax.add_artist(main_legend)
        elif (mode == "scatter"):
            plotting_mode[mode](ax)(indices[idx], val, label=label[idx], alpha=alphas[idx], color=color[idx], **kwargs)

    # Main Legend
    if True:
        handles, labels = ax.get_legend_handles_labels()
        if (handles != []) & (labels != []):
            main_legend = ax.legend(handles, labels, loc=loc_label)
            ax.add_artist(main_legend)

        if showY:
            anchor_y_cumul = anchor_y
            for idx, val in enumerate(values):
                handles_y, labels_y = [], []
                
                # Observed and Predicted : Mean, std, median
                mean_yvalues = val.mean() if not ignore_nan else np.nanmean(val)
                std_yvalues = val.std() if not ignore_nan else np.nanstd(val)
                median_yvalues = np.median(val) if not ignore_nan else np.nanmedian(val)
        
                mean_yvalues_label, mean_yvalues_patch = legend_patch(f"mean = {mean_yvalues:.3f}")
                std_yvalues_label, std_yvalues_patch = legend_patch(f"std = {std_yvalues:.3f}")
                median_yvalues_label, median_yvalues_patch = legend_patch(f"median = {median_yvalues:.3f}")
    
                handles_y.extend([mean_yvalues_patch, std_yvalues_patch, median_yvalues_patch])
                labels_y.extend([mean_yvalues_label, std_yvalues_label, median_yvalues_label])
    
                msm_observed_legend = ax.legend(
                    handles_y, labels_y, title=label[idx], loc="center",
                    handlelength=0, handletextpad=0, borderaxespad=0,
                    bbox_to_anchor=(anchor_x, anchor_y_cumul)
                )
                anchor_y_cumul -= 0.14 if label == ("", ) else 0.17
                ax.add_artist(msm_observed_legend)  # Add legend to artist

        if showR2:  # R2
            handles_r2, labels_r2 = [], []
            for idx, val in enumerate(values):
                isnotnan = ~np.isnan(val)  # index of not NaN values
    
                r2 = r2_score(indices[idx][isnotnan], val[isnotnan])
                r2_label, r2_patch = legend_patch(f"R2 = {r2:.4f}")
                handles_r2.append(r2_patch)
                labels_r2.append(r2_label)

            r2_legend = ax.legend(
                handles_r2, labels_r2, loc=loc_r2,
                handlelength=0, handletextpad=0
            )

            # Add legend
            ax.add_artist(r2_legend)

    # Set scale selected by user
    if scale is not None:
        scale_x = scale
        scale_y = scale

    ax.set_xscale(scale_x)
    ax.set_yscale(scale_y)
    # Option for specific mode
    if (mode == "scatter"):
        # Range to have xlim=ylim
        xy_lim = auxiliary.min_max(ax.get_xlim() + ax.get_ylim())
        ax.set_xlim(xy_lim)
        ax.set_ylim(xy_lim)
    if (mode == "bar"):
        bar_idx_shift = indices[0] * (spacing*n_obs + n_obs/2)
        bar_center_pos = (kwargs["width"] * n_obs * indices[0])
        ax.set_xticks(
            bar_center_pos + bar_idx_shift + indices[0], indices[0]
        )

    # Set figure label, limit and legend    
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    # Label
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Limit in x and y
    ax.set_xlim(left=xleft, right=xright)
    ax.set_ylim(top=ytop, bottom=ybottom)
    # Grid
    ax.grid(grid)

    # Save plot if {save_to} is specified
    if save_to is not None:
        if not auxiliary.isdir(save_to):
            raise Exception("Specified directory does not exists")

        ext = ext.replace(".", "")
        save_to = auxiliary.to_dirpath(save_to)
        filepath = save_to + filename + "." + ext
        if not forcename:
            root, _ = os.path.splitext(filename)
            # Save file to
            filename = auxiliary.replace_extension(root, ext)
            filepath = save_to + filename if overwrite else \
                       auxiliary.filepath_with_suffix(save_to + filename)

        plt.savefig(filepath, bbox_inches='tight')

    return fig, ax


if __name__ == "__main__":
    indices = [0, 1, 2]
    obs = [1, 2, 2]
    pred = [1.1, 2.2, 3.1]
    plot(indices, obs, pred, mode="plot", save_to="./")
