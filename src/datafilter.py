# Data 
import numpy as np
import pandas as pd

# Local modules
import constantes as cst
from auxiliary import read_dataframe
from auxiliary import to_dirpath

# Path to dataframes
filepath_wt = "../data/WTconcatenate.csv.gz"
filepath_ki = "../data/KIconcatenate.csv.gz"

# Filter by these criteria
mask_condition = [cst.WT, cst.KI]
mask_type = [cst.CD3, cst.LY6]
mask_tumor = [cst.IN_TUMOR, cst.OUT_TUMOR]  # Only on or outside tumor
mask_density = [cst.IN_FIBER, cst.OUT_FIBER]  # Only in or outside fiber

apply_rmv_none = False  # True or False
apply_aberrant = -3  # Set to None or actual value

# Defined classes
df_classes = [
    cst.T_PLUS,
    cst.T_ENRICHED,
    cst.T_ENRICHED_2
]


def plus_class_mask(df, colname=cst.CELLS100UM.column):
    """"""
    return df[colname] > 0


def enriched_class_mask(df, colname=cst.CELLS100UM.column, fn=np.mean):
    """"""
    return df[colname] > fn(df[colname])

def enriched_2_class_mask(
    df,
    mask_plus=None,
    colname=cst.CELLS100UM.column,
    fn=np.mean
):
    if mask_plus is None:
        mask_plus = plus_class_mask(df, colname=colname)

    return df[colname] > fn(df[mask_plus][colname])


def add_classes(df):
    """"""
    mask_plus = None
    mask_enriched = None
    mask_enriched_2 = None

    if cst.T_PLUS in df_classes:
        mask_plus = plus_class_mask(df)
        df.loc[:, (cst.T_PLUS.column,)] = 0
        df.loc[mask_plus, (cst.T_PLUS.column,)] = 1

    if cst.T_ENRICHED in df_classes:
        mask_enriched = enriched_class_mask(df)
        df.loc[:, (cst.T_ENRICHED.column,)] = 0
        df.loc[mask_enriched, (cst.T_ENRICHED.column,)] = 1

    if cst.T_ENRICHED_2 in df_classes:
        mask_enriched_2 = enriched_2_class_mask(df, mask_plus=mask_plus)
        df.loc[:, (cst.T_ENRICHED_2.column,)] = 0
        df.loc[mask_enriched_2, (cst.T_ENRICHED_2.column,)] = 1


def cst_mask(df, cst):
    """Apply a mask based on Constantes attributes"""
    return cst == df[cst.column]


def arg_mask(df, arg=None):
    """Returns a mask from the specified filter.

    df: pandas.DataFrame
        pandas dataframe

    arg: pandas.DataFrame or cst.Constantes
        a pandas.DataFrame mask or Constantes class
        with the specified column and value to apply
        mask on.

    Returns: pandas.DataFrame
        The specified mask

    """
    if isinstance(arg, cst.Constantes):
        return cst_mask(df, arg)
    else:
        return arg


def masks_filter(df, *args, filter=np.all, return_mask=False):
    """Apply the specified mask to the dataframe.
    
    df: pandas.DataFrame
        pandas dataframe

    filter: funct
        boolean comparator function

    *args: list of Constantes and/or pandas.DataFrame
        list containing Constantes with the specified
        column and value for dataframe masking, or 
        pandas.DataFrame mask.
    
    return: pandas.DataFrame
        Filtered pandas.DataFrame

    """
    masks = [arg_mask(df, arg) for arg in args]
    combined_mask = filter(masks, axis=0)
    if return_mask:
        return combined_mask

    return df[combined_mask]


def to_filtered_df(df, return_mask=False):
    """"""
    df = df.replace(np.nan, None)  # Replace NaN by None
    filters = [mask_condition, mask_type, mask_tumor, mask_density]
    masks = (
        masks_filter(df, *mask_i, filter=np.any, return_mask=True)
        for mask_i in filters if len(mask_i) > 0
    )
    df_c = masks_filter(df, *masks, filter=np.all, return_mask=return_mask)

    if apply_aberrant is not None:
        df_mask_density = df_c[cst.IN_FIBER.column] > 0
        df_c.loc[~df_mask_density, cst.aberrant_columns] = apply_aberrant

    if apply_rmv_none:
        df_c.dropna(inplace=True)

    # Add specified class
    if len(df_classes) > 0:
        add_classes(df_c)
        
    return df_c


def to_filtered_file(
    df,
    apply_type=cst.data_type,
    dirname=to_dirpath(cst.DATA_DIR)
):
    """"""
    if apply_type is not None:
        df.astype(apply_type)

    if len([
        *mask_condition,
        *mask_type,
        *mask_tumor,
        *mask_density
    ]) == 0:
        df.to_csv(dirname + "ALL.csv")

    filename = ""
    if len(mask_condition) > 0:
        for idx, value in enumerate(mask_condition):
            if (idx == 0):
                filename = filename + value.name
            else:
                filename = filename + "-" + value.name

    if len(mask_type) > 0:
        filename += "_" if len(filename) > 0  else ""
        for idx, value in enumerate(mask_type):
            if (idx == 0):
                filename = filename + value.name
            else:
                filename = filename + "-" + value.name

    if len(mask_tumor) > 0 and all(mask_tumor):
            filename += "_" if len(filename) > 0  else ""
            filename = filename + mask_tumor[0].name

    if len(mask_density) > 0 and all(mask_density):
            filename += "_" if len(filename) > 0  else ""
            filename = filename + mask_density[0].name

    if apply_rmv_none:
        filename = filename + "_" + "no-none"

    if apply_aberrant is not None:
        filename = filename + "_" + "aberrant"

    filepath = dirname + filename + ".csv"

    # To CSV
    df_filtered = to_filtered_df(df)
    df_filtered.to_csv(
        filepath,
        index=0,
        index_label=False
    )

    print(f"File saved at:\n\t{filepath}")


if __name__ == "__main__":
    # Attributes
    mask_condition = [cst.WT, cst.KI]
    mask_type = [cst.LY6]
    mask_tumor = [cst.IN_TUMOR]  # Only on or outside tumor
    mask_density = [cst.IN_FIBER, cst.OUT_FIBER]  # Only in or outside fiber

    apply_rmv_none = False  # True or False
    apply_aberrant = -3  # Set to None or actual value

    # Defined classes
    df_classes = [
        cst.T_PLUS,
        cst.T_ENRICHED,
        cst.T_ENRICHED_2
    ]

    # Read dataframes
    df_wt = None
    df_ki = None
    if cst.WT in mask_condition:
        df_wt = read_dataframe(filepath_wt, low_memory=False)
    if cst.KI in mask_condition:
        df_ki = read_dataframe(filepath_ki, low_memory=False)

    df_all = pd.concat([df_wt, df_ki])

    # Save csv with specified filter
    to_filtered_file(df_all)
