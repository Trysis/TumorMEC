import os

# Data gestion
import numpy as np
import pandas as pd

# Local modules
import constantes as cst
from auxiliary import read_dataframe
from auxiliary import to_dirpath

DIRNAME = "../"

if __name__ == "__main__":
    DIRNAME = "".join(
        os.path.realpath(__file__).split("src/")[:-1]
    ) + "data/"

# Path to dataframes
FILEPATH_WT = DIRNAME + "WTconcatenate.csv.gz"
FILEPATH_KI = DIRNAME + "KIconcatenate.csv.gz"

# Filter by these criteria
MASK_CONDITION = [cst.WT, cst.KI]
MASK_TYPE = [cst.CD3, cst.LY6]
MASK_TUMOR = [cst.IN_TUMOR, cst.OUT_TUMOR]  # Only on or outside tumor
MASK_DENSITY = [cst.IN_FIBER, cst.OUT_FIBER]  # Only in or outside fiber

APPLY_RMV_NONE = False  # True or False
APPLY_ABERRANT = -3  # Set to None or actual value

# Defined classes
DF_CLASS = [
    cst.T_PLUS,
    cst.T_ENRICHED,
    cst.T_ENRICHED_2
]


# Functions
def plus_cmask(df, colname=cst.CELLS100UM.column):
    """Returns a boolean mask with t-plus class condition
    
    df: pandas.DataFrame
        dataframe containing data

    colname: str
        name of the column used for the
        class condition

    Returns: pandas.DataFrame
        Returns a boolean dataframe mask

    """
    return df[colname] > 0


def enriched_cmask(df, colname=cst.CELLS100UM.column, fn=np.mean):
    """Returns a boolean mask with enriched class condition
    
    df: pandas.DataFrame
        dataframe containing data

    colname: str
        name of the column used for the
        class condition

    fn: funct (np.mean or np.median)
        Used function for the class condition,
        user should either use np.mean, np.median
        or a function returning a unique value

    Returns: pandas.DataFrame
        Returns a boolean dataframe mask

    """
    return df[colname] > fn(df[colname])

def enriched_2_cmask(
    df,
    mask_plus=None,
    colname=cst.CELLS100UM.column,
    fn=np.mean
):
    """Returns a boolean mask with enriched_2 class condition

    df: pandas.DataFrame
        dataframe containing data

    mask_plus: pandas.DataFrame
        Pre-computed mask from the data
        for the t-plus class

    colname: str
        name of the column used for the
        class condition

    fn: funct (np.mean or np.median)
        Used function for the class condition,
        user should either use np.mean, np.median
        or a function returning a unique value

    Returns: pandas.DataFrame
        Returns a boolean dataframe mask
    """
    if mask_plus is None:
        mask_plus = plus_cmask(df, colname=colname)

    return df[colname] > fn(df[mask_plus][colname])


def add_classes(df):
    """Add the existing defined class to the dataframe
    
    df:pandas.DataFrame
        dataframe containing data to append
        new class on
    
    Returns: None
        The changes are made in-place

    """
    mask_plus = None
    mask_enriched = None
    mask_enriched_2 = None

    if cst.T_PLUS in DF_CLASS:
        mask_plus = plus_cmask(df)
        df.loc[:, (cst.T_PLUS.column,)] = 0
        df.loc[mask_plus, (cst.T_PLUS.column,)] = 1

    if cst.T_ENRICHED in DF_CLASS:
        mask_enriched = enriched_cmask(df)
        df.loc[:, (cst.T_ENRICHED.column,)] = 0
        df.loc[mask_enriched, (cst.T_ENRICHED.column,)] = 1

    if cst.T_ENRICHED_2 in DF_CLASS:
        mask_enriched_2 = enriched_2_cmask(df, mask_plus=mask_plus)
        df.loc[:, (cst.T_ENRICHED_2.column,)] = 0
        df.loc[mask_enriched_2, (cst.T_ENRICHED_2.column,)] = 1


def cst_mask(df, cst):
    """Apply a mask based on constantes.Constantes attributes"""
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


def masks_filter(df, *args, filter=np.all, return_mask=False, view=False):
    """Apply the specified mask to the dataframe.
    
    df: pandas.DataFrame
        pandas dataframe

    *args: list of Constantes and/or pandas.DataFrame
        list containing Constantes with the specified
        column and value for dataframe masking, or 
        pandas.DataFrame mask.

    filter: function
        boolean list comparator function

    return_mask: bool
        Should we return a mask from the combined
        condition {args} applied on the dataset

    view: bool
        If {return_mask} is False, should we return
        a view of the masked dataset, or return a
        copy ?
 
    return: pandas.DataFrame
        Filtered pandas.DataFrame

    """
    masks = [arg_mask(df, arg) for arg in args]
    combined_mask = filter(masks, axis=0)
    if return_mask:
        return combined_mask

    if view:
        return df[combined_mask]

    return df[combined_mask].copy()


def to_filtered_df(df, return_mask=False):
    """Apply a set of filter pre-selected from the global
    argument, to create a filtered dataframe

    df: pandas.DataFrame
        dataframe containing data

    return_mask: bool
        Should the user return the boolean
        mask or dataframe

    Returns: pandas.DataFrame
        Either the filtered dataframe, or the
        mask with the respective filter

    """
    df = df.replace(np.nan, None)  # Replace NaN by None
    filters = [MASK_CONDITION, MASK_TYPE, MASK_TUMOR, MASK_DENSITY]
    masks = (
        masks_filter(df, *mask_i, filter=np.any, return_mask=True)
        for mask_i in filters if len(mask_i) > 0
    )
    df_c = masks_filter(df, *masks, filter=np.all, return_mask=return_mask)

    if APPLY_ABERRANT is not None:
        df_mask_density = df_c[cst.IN_FIBER.column] > 0
        df_c.loc[~df_mask_density, cst.aberrant_columns] = APPLY_ABERRANT

    if APPLY_RMV_NONE:
        df_c.dropna(inplace=True)

    # Add specified class
    if len(DF_CLASS) > 0:
        add_classes(df_c)
        
    return df_c


def to_filtered_file(
    df,
    apply_type=cst.data_type,
    dirname=to_dirpath(DIRNAME + "data")
):
    """Save dataframe after filtering

    apply_type: dict -> {"column": type, ...}
        dictionnary containing the column
        and the associated type to convert
        the column value into

    dirname: str
        Directory name

    """
    if apply_type is not None:
        df.astype(apply_type)

    if len([
        *MASK_CONDITION,
        *MASK_TYPE,
        *MASK_TUMOR,
        *MASK_DENSITY
    ]) == 0:
        df.to_csv(dirname + "ALL.csv")

    filename = ""
    if len(MASK_CONDITION) > 0:
        for idx, value in enumerate(MASK_CONDITION):
            if (idx == 0):
                filename = filename + value.name
            else:
                filename = filename + "-" + value.name

    if len(MASK_TYPE) > 0:
        filename += "_" if len(filename) > 0  else ""
        for idx, value in enumerate(MASK_TYPE):
            if (idx == 0):
                filename = filename + value.name
            else:
                filename = filename + "-" + value.name

    if len(MASK_TUMOR) > 0 and all(MASK_TUMOR):
            filename += "_" if len(filename) > 0  else ""
            filename = filename + MASK_TUMOR[0].name

    if len(MASK_DENSITY) > 0 and all(MASK_DENSITY):
            filename += "_" if len(filename) > 0  else ""
            filename = filename + MASK_DENSITY[0].name

    if APPLY_RMV_NONE:
        filename = filename + "_" + "no-none"

    if APPLY_ABERRANT is not None:
        # Aberrant exist for None and OUT_FIBER samples
        if not (APPLY_RMV_NONE & (MASK_DENSITY==[cst.IN_FIBER])):
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
    MASK_CONDITION = [cst.WT, cst.KI]
    MASK_TYPE = [cst.CD3]
    MASK_TUMOR = [cst.IN_TUMOR]  # Only on or outside tumor
    MASK_DENSITY = [cst.IN_FIBER]  # Only in or outside fiber

    APPLY_RMV_NONE = True  # True or False
    APPLY_ABERRANT = -3  # Set to None or actual value

    # Defined classes
    DF_CLASS = [
        cst.T_PLUS,
        cst.T_ENRICHED,
        cst.T_ENRICHED_2
    ]

    # Read dataframes
    df_wt, df_ki = None, None
    if cst.WT in MASK_CONDITION:
        df_wt = read_dataframe(FILEPATH_WT, low_memory=False)
    if cst.KI in MASK_CONDITION:
        df_ki = read_dataframe(FILEPATH_KI, low_memory=False)
        df_ki.loc[df_ki["Condition"] == "CD64-hDTR", "Condition"] = "KI"

    df_all = pd.concat([df_wt, df_ki])

    # Save csv with specified filter
    to_filtered_file(df_all)
