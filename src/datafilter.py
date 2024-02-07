# Data 
import numpy as np
import pandas as pd

# Local modules
import constantes
import auxiliary

# Path to dataframes
filepath_wt = "../data/WTconcatenate.csv.gz"
filepath_ki = "../data/KIconcatenate.csv.gz"

mask_condition = [constantes.WT, constantes.KI]
mask_type = [constantes.CD3, constantes.LY6]
mask_tumor = [constantes.IN_TUMOR, constantes.OUT_TUMOR] #0 outside, 1 inside


def arg_mask(df, arg=None):
    """Returns a mask from the specified filter.

    df: pandas.DataFrame
        pandas dataframe

    arg: pandas.DataFrame or constantes.Constantes
        a pandas.DataFrame mask or Constantes class
        with the specified column and value to apply
        mask on.

    Returns: pandas.DataFrame
        The specified mask

    """
    if isinstance(arg, constantes.Constantes):
        return constantes_mask(df, arg)
    else:
        return arg


def constantes_mask(df, cst):
    """Apply a mask based on Constantes attributes"""
    return df[cst.column] == cst.value


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
    mask = filter([arg_mask(df, arg) for arg in args], axis=0)
    if return_mask:
        return mask

    return df[mask]


def to_filtered_df(df, return_mask=False):
    """"""
    masks = [mask_condition, mask_type, mask_tumor]
    masks = [(
        masks_filter(df, *mask_i, filter=np.any, return_mask=True)
        for mask_i in masks if len(mask_i) > 0
        )]
    
    return masks_filter(df, *masks, filter=np.all, return_mask=return_mask)


def to_filtered_file(df):
    if all([len(mask_condition) == 0, len(mask_type) == 0, len(mask_tumor) == 0]):
        df.to_csv("ALL")
    filename = ""
    if len(mask_condition) > 0:
        for idx, value in enumerate(mask_condition):
            if idx == 0:
                filename = filename + value.name
            else:
                filename = filename + "_" + value.name

    if len(mask_type) > 0:
        filename += "_" if filename != "" else ""
        for idx, value in enumerate(mask_type):
            if idx == 0:
                filename = filename + value.name
            else:
                filename = filename + "_" + value.name

    if len(mask_tumor) > 0:
        filename += "_" if filename != "" else ""
        for idx, value in enumerate(mask_tumor):
            if idx == 0:
                filename = filename + value.name
            else:
                filename = filename + "_" + value.name
    
    
    to_filtered_df(df).to_csv(

    )


if __name__ == "__main__":
    # Read dataframes
    df_wt = auxiliary.read_dataframe(filepath_wt, nrows=100000, low_memory=False)
    #df_ki = auxiliary.read_dataframe(filepath_ki, low_memory=False)
    print(df_wt)
    
    print(
        masks_filter(
            df_wt,
            constantes.WT,
            constantes.Constantes("X", 6543)
        )
    )
    # # Column type to each columns
    # data_type = {
    #     **dict.fromkeys(constantes.str_columns, object),
    #     **dict.fromkeys(constantes.unsigned_columns, np.uint32),
    #     **dict.fromkeys(constantes.float_columns, np.float64),
    #     **dict.fromkeys(constantes.int_columns, np.int32)
    # }

    # df_wt = df_wt.astype(data_type)
    # df_ki = df_ki.astype(data_type)

    # # Concatenate both dataset WT & KI
    # df_all = pd.concat([df_wt, df_ki])

    # # T Cells only
    # df_cd3 = df_all[df_all[constantes.CD3[0]] == constantes.CD3[1]]  # T Cells
    # df_cd3_in_tumor = df_cd3[df_cd3[constantes.IN_TUMOR[0]] == constantes.IN_TUMOR[1]]  # T Cells & in tumor
    # df_cd3_out_tumor = df_cd3[df_cd3[constantes.OUT_TUMOR[0]] == constantes.OUT_TUMOR[1]]  # T Cells & outside tumor
