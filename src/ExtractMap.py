#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract a feature as a map from a given dataframe
Exemple: 
    m = ExtractMap(df,"Density20")
    plt.imshow(m)

Created on Wed Sep 7 11:50:57 2022

@author: paolo pierobon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ExtractMap(data, feature_name, chosen_file=None):
    """Extract the feature map containing in a matrix
        the features at the respective position. It is
        used to show regions intensity.

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
        df = data[data["FileName"] == chosen_file]
    #
    n_points = df.shape[0]
    x = np.array((df['X']-df['X'].min())/40, dtype=np.int32)
    y = np.array((df['Y']-df['Y'].min())/40, dtype=np.int32)
    # Matrix of size max(x) * max(y)
    matrix = np.zeros([y.max()+1, x.max()+1])
    # Add features to matrix
    feature_value = np.array(df[feature_name])
    for i in range(n_points):
        matrix[y[i], x[i]]= feature_value[i]

    return matrix
