"""Contains utilitary functions."""

import os

# Pattern matching
import re

import numpy as np
import pandas as pd

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


def create_dir(dirpath, add_suffix=False):
    """Create a directory."""
    if not isdir(os.path.dirname(dirpath)):
        return None

    if add_suffix:
        dirpath = append_suffix(dirpath)

    if os.path.exists(dirpath):
        return dirpath

    os.mkdir(dirpath)
    return dirpath


def read_dataframe(filepath, **kwargs):
    """Read a dataframe."""
    if not isfile(filepath):
        raise Exception("Filepath is invalid")

    return pd.read_csv(filepath, **kwargs)


def min_max_normalization(values, min_scale, max_scale, ignore_nan=True):
    """Normalize values on a specified min and max range.

    values: array-like (numpy.ndarray) -> shape (n_samples, x)
        Values to perform normalization on
    min_scale: float
        Bottom range limit to apply on values so that
        values range from [values.min, values.max] to values[min_scale, values.max]
    max_scale: float
        Upper range limit to apply on values so that
        values range from [values.min, values.max] to values[values.min, max_scale]

    Returns: array-like of shape (n_samples, x)
        Normalized array in range [min_scale, max_scale]

    """
    min_val = values.min() if not ignore_nan else np.nanmin(values)
    max_val = values.max() if not ignore_nan else np.nanmax(values)
    # Normalization
    scale_plage = max_scale - min_scale
    val_plage = max_val - min_val
    flex_shift = values - min_val
    flex_normalized = (flex_shift * (scale_plage/val_plage)) + min_scale

    # Returns
    return flex_normalized


def replace_extension(name, new_ext):
    """Takes a name and replace the existing extension
        by a specified extension. Or simply add the specified
        extension.

    name: str
        Name of the string to add the extension to
    new_ext: str
        Extension value to append to {name}

    Returns: str
        The new name with {name}.{new_ext}

    """
    root, _ = os.path.splitext(name)
    new_ext = new_ext.replace(".", "")
    if new_ext == "":
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


if __name__ == "__main__":
    filepath = "./data/test.py"
    filepath_with_suffix = append_suffix(filepath)
    print(filepath_with_suffix)