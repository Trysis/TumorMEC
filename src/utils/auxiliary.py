"""Contains auxiliary functions."""

import re  # Pattern matching
import os

# Data Gestion
import numpy as np
import pandas as pd

__author__ = "Roude JEAN MARIE"
__email__ = "roude.bioinfo@gmail.com"


def isfile(filepath):
    """Checks if {filepath} is a valid path to file."""
    return os.path.isfile(filepath)


def isdir(dirpath):
    """Checks if {dirpath} is a valid directory."""
    return os.path.isdir(dirpath)


def to_dirpath(dirpath, dir_sep="/"):
    """Returns a {dirpath} with its ending file separator."""
    dirpath = dirpath if dirpath[-1] == dir_sep else \
              dirpath + dir_sep

    return dirpath


def create_dir(dirpath, dir_sep="/", add_suffix=False):
    """Create a directory and return its path."""
    dirpath_ = to_dirpath(dirpath, dir_sep=dir_sep)
    if add_suffix:
        dirpath_ = append_suffix(dirpath_)

    if os.path.exists(dirpath_):
        return dirpath_

    os.mkdir(dirpath_)
    return dirpath_


def read_dataframe(filepath, **kwargs):
    """Read a dataframe."""
    if not isfile(filepath):
        raise Exception("Filepath is invalid")

    return pd.read_csv(filepath, **kwargs)


def min_max(arraylike):
    """Returns the min and max from an array."""
    return min(arraylike), max(arraylike)


def min_max_normalization(values, min_scale=0, max_scale=1, ignore_nan=True):
    """Normalize values on a specified min and max range.

    values: numpy.ndarray -> shape (n_samples, x)
        Values to perform normalization on
    min_scale: float
        Bottom range limit to apply on values so that
        values range from [values.min, values.max] to values[min_scale, values.max]
    max_scale: float
        Upper range limit to apply on values so that
        values range from [values.min, values.max] to values[values.min, max_scale]

    Returns: dict
        values: array-like of shape (n_samples, x)
            Normalized array in range [min_scale, max_scale]
        min: scalar
            minimum value of input values
        max: scaler
            maximum value of inpurt values

    """
    min_val = values.min() if not ignore_nan else np.nanmin(values)
    max_val = values.max() if not ignore_nan else np.nanmax(values)
    # Normalization
    scale_plage = max_scale - min_scale
    val_plage = max_val - min_val
    flex_shift = values - min_val
    flex_normalized = (flex_shift * (scale_plage/val_plage)) + min_scale

    # Returns
    to_return = {
        "values": flex_normalized,
        "min": min_val,
        "max": max_val
    }
    return to_return


def replace_extension(name, new_ext):
    """Takes a name and replace the existing extension
        by a specified extension. Or simply add the
        specified extension.

    name: str
        Name of the string to add the extension to
    new_ext: str
        Extension value to append to {name}

    Returns: str
        The new name with {name}.{new_ext}

    """
    root, _ = os.path.splitext(name)
    new_ext = new_ext.replace(".", "")
    if new_ext in ("", None):
        return root

    name_ext = root + "." + new_ext
    return name_ext


def append_suffix(filepath, path_sep="/", suffix_sep="_"):
    """Takes a path to a file and append a suffix on the file
        name if necessary. It is used in case we want to have
        multiple version of the same file while not removing the
        previous ones.

    filepath: str
        Path to the file
    path_sep: str
        Path character separator. It can differ
        between Linux and Windows, so it should be
        changed accordingly.
    suffix_sep: str
        Character that will separate the actual filename
        from the count value in filepath={dirpath}{filename}
        {suffix_sep}{count}{ext}

    Returns: str
        The new filepath if the file was already existant,
        else it returns filepath.

    """
    # If no existing file to the path exists
    # ,we return the actual path
    if not os.path.exists(filepath):
        return filepath
    filepath = filepath if filepath[-1] != path_sep else filepath[:-1]
    # Else we append a suffix
    dirname = os.path.dirname(filepath)  # directory name
    dirname = "." if dirname == "" else dirname
    dirname = to_dirpath(dirname, path_sep)
    filename = os.path.basename(filepath)  # file name
    file_no_ext, ext = os.path.splitext(filename)
    # Match files with the same pattern as ours
    to_match = file_no_ext + suffix_sep + "[0-9]+" + "[.]" + ext[1:] + "$"
    pattern_file = re.compile(to_match)
    pattern_number = re.compile("\d")
    matched_file = [pattern_file.search(filename) for filename in os.listdir(dirname)]
    matched_file = [found.group(0) for found in matched_file if found is not None]
    # Match with suffix to get the maximum value after {suffix_sep} in matched files
    matched_suffix = [filename[slice(*pattern_number.search(filename).span())] \
                     for filename in matched_file]
    matched_suffix = [0] + [int(number) for number in matched_suffix]
    max_suffix = max(matched_suffix) + 1
    filepath_suffix = f"{dirname}/{file_no_ext}{suffix_sep}{max_suffix}{ext}"

    return filepath_suffix


def get_metrics(array, q=(0, 0.25, 0.5, 0.75, 1), ignore_nan=False):
    """This function returns a panel of statistics calculated on the
        provided {array} argument.

    array: array-like (list, tuple, numpy.ndarray, ...)
        list containing the values to mesure metrics on

    q: array-like
        probability for the quantiles, such that all
        values are comprised between [0, 1]

    ignore_nan: bool
        Should we ignore nan values during the 
        computation ?

    Returns: dict
        Dictionnary containing the different calculated
        metrics of {values}. Such as mean, std, median,
        and quantile.

    """
    if not isinstance(array, np.ndarray):
        try:
            array = np.array(array)
        except:
            raise Exception("{array} cannot be converted to np.ndarray")

    # Array metric
    arr_mean = array.mean() if not ignore_nan else np.nanmean(array)
    arr_sd = array.std() if not ignore_nan else np.nanstd(array) 
    arr_median = np.median(array) if not ignore_nan else np.nanmedian(array)
    arr_quantile = np.quantile(array, q) if not ignore_nan else np.nanquantile(array, q)

    to_return = {
        "mean": arr_mean,
        "std": arr_sd,
        "median": arr_median,
        "quantile": arr_quantile
    }

    return to_return


def format_by_rows(array, ncol=1, spacing=3):
    """Returns values aranged in a specified format.
    
    array: array like
        a list of values to display, such as str,
        int, float or even bool

    ncol: int
        number of column for the formating

    spacing: int
        space between each array value

    Returns: str
        A string representation of the values
        from {array} as specified by the argument

    """
    arr_str = [f"{value}" for value in array]
    arr_len = [len(value) for value in arr_str]
    max_len = max(arr_len)
    # format
    val_format = f"<{max_len}s"
    spacing_format = f"{spacing}s"
    # to_return
    val_str = [
        f"{val:{val_format}}{'':{spacing_format}}" if (idx+1)%ncol != 0\
        else f"{val:{val_format}}{'':{spacing_format}}\n"
        for idx, val in enumerate(arr_str)
    ]
    val_str = "".join(val_str)

    return val_str


if __name__ == "__main__":
    filepath = "./data/test.py"
    filepath_with_suffix = append_suffix(filepath)
    print(filepath_with_suffix)
