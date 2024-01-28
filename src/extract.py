#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contains function that works on the studied tumor dataset."""

import numpy as np

# Local modules
import auxiliary
import plots

def ExtractMap(
    data, feature_name, chosen_file=None,
    name_column="FileName", x_column="X", y_column="Y"
):
    """Extract the feature map containing in a matrix
        the features at the respective position. It is
        used to show regions intensity.

    -----
    Extract a feature as a map from a given dataframe
    Exemple: 
        m = ExtractMap(df,"Density20")
        plt.imshow(m)
    
    Created on Wed Sep 7 11:50:57 2022
    @author: paolo pierobon

    -----
    data: pandas.Dataframe
        A dataframe formatted such that it contains
        an X and Y column corresponding to positions
        where a specfic observation has been made.

    chosen_file: str
        In case if dataframe is formatted such that

    Returns: np.ndarray
        The matrix containing features values
        associated to each position.

    """
    df = data
    if isinstance(chosen_file, str):
        df = data[data[name_column] == chosen_file]

    # 
    n_points = df.shape[0]
    x = np.array((df[x_column]-df[x_column].min())/40, dtype=np.int32)
    y = np.array((df[y_column]-df[y_column].min())/40, dtype=np.int32)

    # Matrix of size max(x) * max(y)
    matrix = np.zeros([y.max()+1, x.max()+1])
    # Add features to matrix
    feature_value = np.array(df[feature_name])
    for i in range(n_points):
        matrix[y[i], x[i]]= feature_value[i]

    return matrix


def extract_cells_metrics(
    data, feature_name, obs_names=None,
    name_column="FileName", **kwargs
):
    """"""
    # Selection of observation to analyse on
    on_files = None
    if obs_names is not None:
        if isinstance(obs_names, str):
            # Perform the mean, std, etc computation
            # on the chosen file.
            on_files = [obs_names]
        elif isinstance(obs_names, (list, tuple, np.ndarray)):
            if all(isinstance(filename, str) for name in obs_names):
                # Perform the same calcul on multiple filename.
                on_files = list(obs_names)
        else:
            raise Exception(
                "If {obs_names} is specified, it should be "
                "a filename (of type str) or a list of filename "
                "of type (list(str))."
            )
    else:  # Else perform on all cells (indicated by {name_column} column)
        on_files = data[name_column].unique()

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
    for filename in on_files:
        cells_metrics_dict[filename] = dict()
        data_file = data[data[name_column] == filename]
        for feature in features:
            cells_metrics_dict[filename][feature] = auxiliary.get_metrics(data_file[feature], **kwargs)

    return cells_metrics_dict


def gen_cells_distribution(
    data, feature_name, obs_names=None, name_column="FileName",
    **kwargs
):
    """"""
    # List of files to plot features on
    # Selection of observation to analyse on
    on_files = None
    if obs_names is not None:
        if isinstance(obs_names, str):
            # Perform the mean, std, etc computation
            # on the chosen file.
            on_files = [obs_names]
        elif isinstance(obs_names, (list, tuple, np.ndarray)):
            if all(isinstance(filename, str) for name in obs_names):
                # Perform the same calcul on multiple filename.
                on_files = list(obs_names)
        else:
            raise Exception(
                "If {obs_names} is specified, it should be "
                "a filename (of type str) or a list of filename "
                "of type (list(str))."
            )
    else:  # Else perform on all cells (indicated by {name_column} column)
        on_files = data[name_column].unique()

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
            fig, ax = plots.plot(
                data[data[name_column] == filename][feature],
                title=title_plt,
                **kwargs
            )
            yield fig


def plot_cells_distribution(
    data,
    feature_name,
    obs_names=None,
    name_column="FileName",
    filepath=None,
    **kwargs
):
    """"""
    then_close = kwargs.pop("then_close") if "then_close" in kwargs else True
    bbox_inches = kwargs.pop("bbox_inches") if "bbox_inches" in kwargs else None
    gen_figures = gen_cells_distribution(
        data=data,
        feature_name=feature_name,
        obs_names=obs_names,
        name_column=name_column,
        **kwargs
    )
    # Plot
    plots.to_pdf(
        filepath,
        gen_figures,
        then_close=then_close,
        bbox_inches=bbox_inches
    )


if __name__ == "__main__":
    pass
