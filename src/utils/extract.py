#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contains function that works on the studied tumor dataset."""

import numpy as np

# Local modules
from src.utils import auxiliary
from src.utils import plots


__author__ = "Roude JEAN MARIE"
__email__ = "roude.bioinfo@gmail.com"


def extract_map(
    data, feature_name, sample_name=None,
    sample_column="FileName", x_column="X", y_column="Y",
    scale=40, coord_type=np.int32
):
    """Created on Wed Sep 7 11:50:57 2022
    @author: paolo pierobon
    
    Extract the feature map containing in a matrix
        the features at the respective position. It is
        used to show regions intensity.

    -----
    Extract a feature as a map from a given dataframe
    Exemple:
        # With df a dataframe with features from tumor images
        m = extract_map(df, "Density20")
        plt.imshow(m)

    -----
    data: pandas.Dataframe
        A dataframe formatted such that it contains
        an X and Y column corresponding to positions
        where a specfic observation has been made.

    chosen_file: str
        In case if dataframe is formatted such that

    sample_column: str
        Column of the dataframe specifiying filenames
        attributed to each sample

    x_column: str
        Column corresponding to the x coordinates

    y_column: str
        Column corresponding to the y coordinates

    scale: int or float
        Scale of the observation in um (micro-meters)

    coord_type: numpy.dtype, default=numpy.int32
        Type attributed to each x and y coordinates

    Returns: np.ndarray
        The matrix containing features values
        associated to each position.

    """
    # Input data
    df = data
    # Change to chosen image sample, if specified
    if isinstance(sample_name, str):
        df = data[data[sample_column] == sample_name]

    # Number of samples
    n_points = df.shape[0]
    x = np.array((df[x_column]-df[x_column].min())/scale, dtype=coord_type)
    y = np.array((df[y_column]-df[y_column].min())/scale, dtype=coord_type)

    # Matrix of size max(x) * max(y)
    matrix = np.zeros([y.max()+1, x.max()+1])
    # Add features to matrix
    feature_value = np.array(df[feature_name])
    for i in range(n_points):
        matrix[y[i], x[i]]= feature_value[i]

    return matrix


def extract_cells_metrics(
    data, feature_name, sample_name=None,
    sample_column="FileName", **kwargs
):
    """Returns the calculated metrics with the specified
        sample and feature names

    data: pandas.Dataframe
        A dataframe formatted such that it contains
        an X and Y column corresponding to positions
        where a specfic observation has been made.

    feature_name: str, or list -> list(str)
        Name(s) of feature(s) to evaluate metrics
        on

    sample_name: str, or list -> list(str)
        Specified sample name to apply statistics
        on, on {sample_column} column name.
        If None, the same calculation is performed
        on each unique sample (previously identified
        with {sample_column}).

    sample_column: str, default="FileName"
        Column of the dataframe specifiying filenames
        attributed to each sample. If None, then the
        metrics are calculated on every observations.

    kwargs: 
        argument to provide to auxiliary.get_metrics

    Returns: dict
        Dictionnary containing for the specified
        samples, the metrics values for each specified
        features

    """
    # Selection of observation to perform analyse on
    on_files = None
    if sample_name is not None:
        # Perform the mean, std, etc computation
        # on the chosen file.
        if isinstance(sample_name, str):
            on_files = [sample_name]
        # Perform the same tasks on multiple filename.
        elif isinstance(sample_name, (list, tuple, np.ndarray)):
            if all(isinstance(name, str) for name in sample_name):
                on_files = list(sample_name)
        else:
            raise Exception(
                "If {obs_names} is specified, it should be "
                "a filename (of type str) or a list of filename "
                "of type (list(str))."
            )
    # Else perform on all cells (indicated by {sample_column} column)
    else:
        if sample_column is not None:
            on_files = data[sample_column].unique()

    # Selection of features
    features = None
    if isinstance(feature_name, str):
        features = [feature_name]
    elif isinstance(feature_name, (list, tuple, np.ndarray)):
        features = list(feature_name)
    else:
        raise Exception(
            "{feature_name} should be an str or "
            "a list of features name (list(str))"
        )

    # Dictionary that will contain all the data
    cells_metrics_dict = dict()
    # Measure only metrics on each observation
    if sample_column is not None:
        for feature in features:
            cells_metrics_dict[feature] = \
                auxiliary.get_metrics(data_file[feature], **kwargs)
    # Measure metrics for each sample (filename), as specified
    else:
        # For each sample
        for filename in on_files:
            cells_metrics_dict[filename] = dict()
            data_file = data[data[sample_column] == filename]
            # Each feature
            for feature in features:
                # Metrics are calculated
                cells_metrics_dict[filename][feature] = \
                    auxiliary.get_metrics(data_file[feature], **kwargs)

    return cells_metrics_dict


def generator_plots_cells(
    data, feature_name, sample_name=None,
    sample_column="FileName", **kwargs
):
    """Generate matplotlib.figure.Figure to plot

    data: pandas.Dataframe
        A dataframe formatted such that it contains
        an X and Y column corresponding to positions
        where a specfic observation has been made.

    feature_name: str, or list -> list(str)
        Name(s) of feature(s) to evaluate metrics
        on

    sample_name: str, or list -> list(str)
        Specified sample name to apply statistics
        on, on {sample_column} column name.
        If None, the same calculation is performed
        on each unique sample (previously identified
        with {sample_column}).

    sample_column: str, default="FileName"
        Column of the dataframe specifiying filenames
        attributed to each sample. If None, then the
        metrics are calculated on every observations.

    kwargs:
        metrics to provide to plots.plot

    Yields: matplotlib.figure.Figure
        A Figure to be plotted, according to the
        different sample and feature name specified

    """
    # List of files to plot features on
    # Selection of observation to perform analyse on
    on_files = None
    if sample_name is not None:
        # Perform the mean, std, etc computation
        # on the chosen file.
        if isinstance(sample_name, str):
            on_files = [sample_name]
        # Perform the same tasks on multiple filename.
        elif isinstance(sample_name, (list, tuple, np.ndarray)):
            if all(isinstance(name, str) for name in sample_name):
                on_files = list(sample_name)
        else:
            raise Exception(
                "If {obs_names} is specified, it should be "
                "a filename (of type str) or a list of filename "
                "of type (list(str))."
            )
    # Else perform on all cells (indicated by {sample_column} column)
    else:
        if sample_column is not None:
            on_files = data[sample_column].unique()

    # Selection of features
    features = None
    if isinstance(feature_name, str):
        features = [feature_name]
    elif isinstance(feature_name, (list, tuple, np.ndarray)):
        features = list(feature_name)
    else:
        raise Exception(
            "{feature_name} should be an str or "
            "a list of features name (list(str))"
        )

    # Plot
    for filename in on_files:
        for feature in features:
            title_plt = f"{filename}\n{feature} Distribution"
            fig, _ = plots.plot(
                data[data[sample_column] == filename][feature],
                title=title_plt,
                **kwargs
            )
            yield fig


def plot_cells(
    data, feature_name, sample_name=None,
    sample_column="FileName", filepath=None,
    **kwargs
):
    """Save distribution plots from the dataframe,
    with the specified sample and feature names.
    Uses generator_plot_cells.

    data: pandas.Dataframe
        A dataframe formatted such that it contains
        an X and Y column corresponding to positions
        where a specfic observation has been made.

    feature_name: str, or list -> list(str)
        Name(s) of feature(s) to evaluate metrics
        on

    sample_name: str, or list -> list(str)
        Specified sample name to apply statistics
        on, on {sample_column} column name.
        If None, the same calculation is performed
        on each unique sample (previously identified
        with {sample_column}).

    sample_column: str, default="FileName"
        Column of the dataframe specifiying filenames
        attributed to each sample. If None, then the
        metrics are calculated on every observations.

    filepath: str
        Path to save pdf containing the plots

    kwargs:
        key=value arguments to provide to plots.plot
        from generator_plot_cells, and plots.to_pdf

    """
    then_close = kwargs.pop("then_close") if "then_close" in kwargs else True
    bbox_inches = kwargs.pop("bbox_inches") if "bbox_inches" in kwargs else None
    gen_figures = generator_plots_cells(
        data=data,
        feature_name=feature_name,
        sample_name=sample_name,
        sample_column=sample_column,
        **kwargs
    )
    # Save plots to pdf
    plots.to_pdf(
        filepath=filepath,
        figures=gen_figures,
        then_close=then_close,
        bbox_inches=bbox_inches
    )


if __name__ == "__main__":
    pass
