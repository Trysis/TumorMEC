import os

# Data gestion
import numpy as np
import pandas as pd

__author__ = "Roude JEAN MARIE"
__email__ = "roude.bioinfo@gmail.com"


class DataLoader:    
    def __init__(
            self, data_dir,
            mask_condition, mask_type,
            mask_tumor, mask_fiber,
            replace_aberrant, remove_none
    ):
        """
        data_dir: str
            Directory containing the data

        mask_condition, mask_type, mask_tumor, mask_fiber: list
            list of Constantes and/or pandas.DataFrame.
            It contains Constantes object with the specified
            column and value for dataframe masking, or an 
            actual pandas.DataFrame mask.

        replace_aberrant: None, int or float
            If specified, aberrant values associated with
            a predefined set of columns for samples in region
            without fiber will have their value set to
            {replace_aberrant}

        remove_none: bool
            Should we remove NaN and None value after
            filtering ?

        """
        self.data_dir = data_dir
        self.mask_condition = mask_condition
        self.mask_type = mask_type
        self.mask_tumor = mask_tumor
        self.mask_fiber = mask_fiber
        self.replace_aberrant = replace_aberrant
        self.remove_none = remove_none
        self.__set_masks__()
        self.__check_attr__()

    def __set_masks__(self):
        masks = [
            self.mask_condition, self.mask_type,
            self.mask_tumor, self.mask_fiber
        ]
        self.masks = [mask for mask in masks
        if mask is not None and len(mask) > 0]

    def __check_attr__(self):
        # Default value for attributes
        if self.data_dir == "":
            self.data_dir = "./"
        if self.mask_condition is None:
            self.mask_condition = list()
        if self.mask_type is None:
            self.mask_type = list()
        if self.mask_tumor is None:
            self.mask_tumor = list()
        if self.mask_fiber is None:
            self.mask_fiber = list()

        # Exception
        if not os.isdir(self.datapath):
            raise Exception("Invalid datapath")
        if not isinstance(self.remove_none, bool):
            raise Exception("remove_none attribute should be a bool")

    def load_data(
            self,
            default_file=[["WTconcatenate.csv.gz", "WT"], ["KIconcatenate.csv.gz", "KI"]],
            targets=None, type=None, save=True, verbose=1, **kwargs
    ):
        """Load an existing dataframe if a file with the specified mask,
        already exist. Else generate a dataframe filtered with the specified
        mask. The given filename is attributed based on mask.

        default_file: list(array-like)
            In case a dataframe with the specified masks is not found,
            this argument serve as the default files to retrieve data on.
            It correspond to a list of list such that each index in
            {default_file} contain a list of min len=2 containing
            [filename, condition, separator].
            Described as such:
                {filename}: str
                    An actual path to a file or
                    path to {self.data_dir}/{filename}
                {condition}: str, or list(str)
                    The specified condition phenotype, such
                    as "WT", "KI", ["WT", "KI"] for both, or other.
                {separator}: str, optional, default=","
                    The separator for the specified file to read
                    with {filename}. By default the separator is ","

        targets: list -> list(callable, dict, pandas.DataFrame), default=None
            This argument is used to add the specified target variable
            attributed to each sample of the original dataframe before
            filtering.
            It can contains a callable taking as input a pandas.DataFrame
            and returning a dictionnary with key as column names and values
            as target values such as {"t_plus": [0, 1, 0, 0, ...]}. Or the
            callable can return a pandas.DataFrame to concatenate with the
            original dataframe.
            It can takes as input a dictionnary with the format as specified
            above, or a pandas.DataFrame.

        type: dict
            Dict with key as column from the dataframe, and
            values as the specified type to convert column to.

        save: bool or str
            If True, it save the filtered dataframe to {self.data_dir}
            If an str is specified, it saves the file to the specified
            directory.
            In False, or unexisting directory the dataframe won't be saved

        verbose: int
            When > 0, it prints some information

        **kwargs:
            Supplementary key argument to pass to callable functions

        Returns: pandas.DataFrame
            The filtered pandas.DataFrame containing the target features
            as specified by the user.

        """
        # Path to a potential existing file
        dirpath = self.data_dir
        filename = self.filename_from_mask(self)
        filepath = os.path.join(dirpath, filename)
        dataframe = None
        if os.path.isfile(filepath):
            dataframe = pd.read_csv(filepath)

        # Read 'default'
        dframes = []
        for lf in default_file:
            if dataframe is not None: break
            lf.append(",")
            filename, condition, separator = lf[0], lf[1], lf[2]
            if (len(self.mask_condition) == 0) or (set(condition).intersection(set(self.mask_condition))):
                filepath = filename if os.path.isfile(filename) else os.path.join(self.data_dir, filename)
                if os.path.isfile(filepath):
                    df_tmp = pd.read_csv(filename, sep=separator, low_memory=False)
                    if condition == "KI":  # replace 'CD64-hDTR' with 'KI'
                        df_tmp.loc[df_tmp["Condition"] == "CD64-hDTR", "Condition"] = "KI" 
                    dframes.append(df_tmp)
                else:
                    raise Exception(f"In default_file: filename={filename} not found\n"
                                    "You should either specify a filename existing "
                                    "in {data_dir} or a valid path to a file")

        # Concatenate the specified files
        dataframe = pd.concat(dframes) if dataframe is None else dataframe
        # Apply types
        if type is not None:
            dataframe = dataframe.astype(type)

        # Add target features
        if targets is not None:
            for cls in targets:
                to_target = cls
                if callable(to_target):
                    to_target = cls(dataframe, **kwargs)
                if isinstance(to_target, dict):
                    to_target = pd.DataFrame(to_target)
                    if to_target.shape[0] != dataframe.shape[0]:
                        raise Exception("Target column does not have the same"
                                        "number of sample as unfiltered concatenated "
                                        f"dataframe n_sample={dataframe.shape[0]};"
                                        f"target n_sample={to_target.shape[0]}")
                elif isinstance(to_target, pd.DataFrame):
                    if to_target.shape[0] != dataframe.shape[0]:
                        raise Exception("Provided dataframe argument does not have the "
                                        "same number of sample as original dataframe\n"
                                        f"\tn_sample={dataframe.shape[0]};"
                                        f"\ttarget n_sample={to_target.shape[0]}")
                # Update and/or Concatenation
                set_target, set_dataframe = set(to_target), set(dataframe.columns)
                same_columns = list(set_target.intersection(set_dataframe))
                diff_columns = list(set_target.difference(set_dataframe))
                if same_columns:
                    dataframe.update(to_target[same_columns])
                if diff_columns:
                    dataframe = pd.concat([dataframe, to_target[diff_columns]], axis=1)

        # Filter data based on mask input
        dataframe = filter_data(
            dataframe, self.masks,
            replace_aberrant=self.replace_aberrant,
            remove_none=self.remove_none
        )

        # Save dataframe
        if isinstance(save, (bool, str)):
            dirpath = self.data_dir if not isinstance(save, str) else save
            if os.path.isdir(dirpath):
                filepath = os.path.join(dirpath, filename)
                dataframe.to_csv(filepath, header=True, index=False)
                if verbose:
                    print(f"File saved at:\n\t{filepath}")
            else:
                if verbose:
                    print("Could not save dataframe, the specified "
                          f"directory does not exist {dirpath}")

        # Assign attributes
        self.loaded = {
            "dirpath": dirpath,
            "filepath": filepath,
            "data": dataframe
        }

        return dataframe

    def filename_from_mask(self):
        """Return a name of file defined by the applied mask"""
        if not hasattr(self, "masks"):
            self.__set_masks__()

        to_add = []
        filename = ""
        if len(self.masks) == 0:
            to_add.append("ALL")
        else:
            if len(self.mask_condition) > 0:
                to_add.append("-".join([c.value for c in self.mask_condition]))
            if len(self.mask_type) > 0:
                to_add.append("-".join([c.value for c in self.mask_type]))
            if len(self.mask_tumor) > 0:
                # all True if duplicate of IN_TUMOR; False with OUT_TUMOR
                # if both we does not need to change filename
                if all(self.mask_tumor) or not any(self.mask_tumor):
                    to_add.append(self.mask_tumor[0].name)
            if len(self.mask_fiber) > 0:
                if all(self.mask_fiber) or not any(self.mask_fiber):
                    to_add.append(self.mask_fiber[0].name)

        if self.replace_aberrant is not None:
            # Aberrant value exists for None and OUT_FIBER samples
            if len(self.mask_fiber) == 0 or (False in self.mask_fiber):
                to_add.append("aberrant")

        if not self.remove_none:
            to_add.append("na")

        filename = "_".join(to_add) + ".csv"
        return filename


def filter_data(
        df, masks, return_mask=False, replace_aberrant=None, remove_none=False,
        aberrant_column=None, fiber_column="Density20"
):
    """Apply a set of filter pre-selected from the specified
    masks argument, to produce a filtered and processed dataframe

    df: pandas.DataFrame
        dataframe containing data

    masks: list of list
        list of list of mask to filter the data
        such as [[cst.WT, cst.KI], [cst.IN_TUMOR]]

    return_mask: bool
        Should the user return the boolean
        mask or dataframe

    replace_aberrant: None or float
        Replace values for regions without fiber when
        a value has been assigned by the specified
        value.

    remove_none: bool
        Should we drop na values in the dataframe ?

    aberrant_column: list(str)
        columns name associated with aberrant values
        for a sample without fiber

    fiber_column: str or list(str)
        column name specifying fiber column, known
        as Density20

    Returns: pandas.DataFramfilterse
        Either the filtered dataframe, or the
        mask with the respective filter

    """
    # Replace NaN by None
    df = df.replace(np.nan, None)
    boolean_masks = (
        masks_filter(df, *mask_i, filter=np.any, return_mask=True)
        for mask_i in masks if len(mask_i) > 0
    )
    df_filtered = masks_filter(df, *boolean_masks, filter=np.all, return_mask=return_mask)

    if replace_aberrant is not None:
        df_mask_density = df_filtered[fiber_column] > 0
        df_filtered.loc[~df_mask_density, aberrant_column] = replace_aberrant

    if remove_none:
        df_filtered.dropna(inplace=True)
        
    return df_filtered


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
        return arg == df[arg.column]
    else:
        return arg


# Functions
def plus_cmask(df, key="t_plus", colname="Cells100um", return_mask=False):
    """Returns a boolean mask with t-plus class condition
    
    df: pandas.DataFrame
        dataframe containing data

    key: str
        Dictionnary key name

    colname: str
        name of the column used for the
        class condition

    return_mask: bool
        If True return a mask associated with the target
        class condition

    Returns: dict
        A dictionnary with key as column name,
        and values as target feature values

    """
    mask = df[[colname]] > 0
    if return_mask:
        return mask
    values = mask.replace({True: 1, False: 0})
    return {key: values.to_numpy()}


def enrich_cmask(
        df, key="t_enrich", colname="Cells100um",
        fn=np.mean, return_mask=False
    ):
    """Returns a boolean mask with enriched class condition
    
    df: pandas.DataFrame
        dataframe containing data

    key: str
        Dictionnary key name

    colname: str
        name of the column used for the
        class condition

    fn: funct (np.mean or np.median)
        Used function for the class condition,
        user should either use np.mean, np.median
        or a function returning a unique value

    return_mask: bool
        If True return a mask associated with the target
        class condition

    Returns: dict
        A dictionnary with key as column name,
        and values as target feature values

    """
    mask = df[[colname]] > fn(df[[colname]])
    if return_mask:
        return mask
    values = mask.replace({True: 1, False: 0})
    return {key: values.to_numpy()}

def enrich_2_cmask(
    df, key="t_enrich_2", mask_plus=None, colname="Cells100um",
    fn=np.mean, return_mask=False
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

    return_mask: bool
        If True return a mask associated with the target
        class condition

    Returns: pandas.DataFrame
        Returns a boolean dataframe mask

    """
    if mask_plus is None:
        mask_plus = plus_cmask(df, colname=colname, return_mask=True)
    mask = df[[colname]] > fn(df[mask_plus][colname])
    if return_mask:
        return mask
    values = mask.replace({True: 1, False: 0})
    return {key: values.to_numpy()}


if __name__ == "__main__":
    """
    MASK_CONDITION = [cst.WT, cst.KI]
    MASK_TYPE = [cst.CD3, cst.LY6]
    MASK_TUMOR = [cst.IN_TUMOR, cst.OUT_TUMOR]  # Only on or outside tumor
    MASK_DENSITY = [cst.IN_FIBER, cst.OUT_FIBER]  # Only in or outside fiber

    APPLY_RMV_NONE = False  # True or False
    APPLY_ABERRANT = -3  # Set to None or actual value
    """
    """
    # Attributes
    MASK_CONDITION = [cst.WT, cst.KI]
    MASK_TYPE = [cst.CD3]
    MASK_TUMOR = [cst.IN_TUMOR]  # Only on or outside tumor
    MASK_DENSITY = [cst.IN_FIBER]  # Only in or outside fiber

    APPLY_RMV_NONE = True  # True or False
    APPLY_ABERRANT = -3  # Set to None or actual value

    loader = DataLoader(
        data_dir="../../data/",
        mask_condition=MASK_CONDITION,
        mask_type=MASK_TYPE,
        mask_tumor=MASK_TUMOR,
        mask_fiber=MASK_DENSITY,
        replace_aberrant=APPLY_ABERRANT,
        remove_none=APPLY_RMV_NONE
    )
    """    
