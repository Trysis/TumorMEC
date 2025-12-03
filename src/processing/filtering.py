# Internal modules
from typing import List, Sequence, Literal, Dict, Union, Optional, Any

# External modules
import pandas as pd
import numpy as np

# User modules

class Preprocess:
    """Preprocessing class to apply on pandas dataframe."""

    def __init__(
        self,
        rows: Optional[Sequence[Union[int, bool]]] = None,
        rows_categories: Optional[Dict[str, Sequence]] = None,
        rows_nonzeros: bool = False,
        rows_nonnan: Union[bool, Sequence[str]] = False,
        cols: Optional[Sequence[Union[int, bool, str]]] = None,
        cols_nonzeros: bool = False,
        cols_nonnan: bool = False,
        permissive: bool = False,
    ):
        """Set the preprocessing attribute to apply.

        rows: list of int or boolean array, optional
            Selected rows on which we apply the filtering
        rows_categories: dict, optional
            When specified, for column in dataframe.columns specified
            by key=feature_name we only keep rows where the feature_name the specified columns (values) in
            key: values
        row_nonzeros: bool, default = True
            Remove empty rows after filtering steps which subset the features
        rows_nonnan: bool or Sequence of str, default=False
            TODO
        cols: list of int, boolean array, or column names, optional
            Selected columns of the dataframe on which we apply the filtering
            A list of int correspond to the column positions [0, n_columns[
            A boolean array should have the same length as the number of cols
            A list of str should have correct column names
        cols_nonzeros: bool, optional
            When specified then empty columns (==0) will be filtered out
        cols_nonnan: bool
            TODO
        permissive: bool
            When set to False, raise an Exception when a key column in
            {col_categories}, are not found in the data during preprocessing
            Also when a column name in cols is not found in the dataframe

        Note:
            {col_categories} should correspond to a dict formatted with keys
            as column name of a dataframe and values as selected categories
            in the column. dict={"column": [category1, category2, ..], ...}.

        """
        # Set attributes (overall order of the preprocessing step)
        self.rows = rows
        self.rows_categories = rows_categories
        self.rows_nonzeros = rows_nonzeros
        self.rows_nonnan = rows_nonnan
        self.cols = cols
        self.cols_nonzeros = cols_nonzeros
        self.cols_nonnan = cols_nonnan
        self.permissive = permissive

    def __call__(
        self,
        dataframe: pd.DataFrame,
        rows: Optional[Sequence[Union[int, bool]]] = None,
        rows_categories: Optional[Dict[str, Sequence]] = None,
        rows_nonzeros: Optional[bool] = None,
        rows_nonnan: Optional[Union[bool, Sequence[str]]] = None,
        cols: Optional[Sequence[Union[int, bool, str]]] = None,
        cols_nonzeros: Optional[bool] = None,
        cols_nonnan: Optional[bool] = None,
        permissive: Optional[bool] = None,
    ):
        """On call, apply the preprocessing on the data."""
        # Check instance
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("{data} should be a pd.DataFrame object")
        # Retrieve specified arguments
        rows = self.rows if rows is None else rows
        rows_categories = self.rows_categories if rows_categories is None else rows_categories
        rows_nonzeros = self.rows_nonzeros if rows_nonzeros is None else rows_nonzeros
        rows_nonnan = self.rows_nonnan if rows_nonnan is None else rows_nonnan
        cols = self.cols if cols is None else cols
        cols_nonzeros = self.cols_nonzeros if cols_nonzeros is None else cols_nonzeros
        cols_nonnan = self.cols_nonnan if cols_nonnan is None else cols_nonnan
        permissive = self.permissive if permissive is None else permissive
        # Mask for samples and features
        sample_mask = (
            np.ones(dataframe.shape[0], dtype=bool)
            & self.mask_rows(dataframe, rows=rows)
            & self.mask_categories(dataframe, rows_categories, permissive)
            & self.mask_rows_nonzeros(dataframe, rows_nonzeros)
        )
        feature_mask = (
            np.ones(dataframe.shape[1], dtype=bool)
            & self.mask_cols(dataframe, cols=cols)
            & self.mask_cols_nonzeros(dataframe, cols_nonzeros)
        )
        # Apply filter
        dataframe = dataframe.loc[sample_mask, feature_mask]
        dataframe = dataframe.loc[
            self.mask_rows_nonnan(dataframe, rows_nonnan),
            self.mask_cols_nonnan(dataframe, cols_nonnan)
        ].copy()
        return dataframe

    @staticmethod
    def mask_rows(
        x: Union[pd.DataFrame, np.ndarray],
        rows: Optional[Sequence[Union[int, bool]]]
    ) -> np.ndarray:
        """Apply the mask on the rows of {x}."""
        # When rows is None, no need to subset by rows
        if rows is None:
            return np.ones(x.shape[0], dtype=bool)
        return select_cols(rows, x.shape[0], as_bool=True)

    @staticmethod
    def mask_cols(
        x: Union[pd.DataFrame, np.ndarray],
        cols: Optional[Sequence[Union[int, bool, str]]]
    ) -> np.ndarray:
        """Apply the mask on the columns of {x}."""
        # When cols is None, no need to subset by cols
        if cols is None:
            return np.ones(x.shape[1], dtype=bool)
        # When cols are specified column names
        if isinstance(cols, str) or isinstance(cols[0], str):
            if isinstance(x, pd.DataFrame):
                return select_colnames(x.columns, cols, as_bool=True)
            else:
                raise TypeError(
                    "x should be a pd.DataFrame when cols contain str"
                )
        return select_cols(cols, x.shape[1], as_bool=True)

    @staticmethod
    def mask_rows_nonzeros(
        x: Union[np.ndarray, pd.DataFrame],
        non_zeros: Optional[bool] = True
    ):
        """Retain non-zero columns."""
        # When non_zeros is False or not set, not necessary to compute
        if non_zeros is None or non_zeros == False:
            return np.ones(x.shape[0], dtype=bool)
        return select_nonzero_rows(x, as_bool=True)

    @staticmethod
    def mask_rows_nonnan(
        x: Union[np.ndarray, pd.DataFrame],
        non_nan: Optional[Union[bool, Sequence[str]]] = True
    ):
        """Retain non-nan rows (can be non-nan of specific columns)."""
        # When non_nan is False or not set, not necessary to compute
        if non_nan is None or non_nan == False:
            return np.ones(x.shape[0], dtype=bool)
        return select_nonnan_rows(x, non_nan=non_nan, as_bool=True)

    @staticmethod
    def mask_cols_nonzeros(
        x: Union[np.ndarray, pd.DataFrame],
        non_zeros: Optional[bool] = True
    ):
        """Retain non-zero columns."""
        # When non_zeros is False or not set, not necessary to compute
        if non_zeros is None or non_zeros == False:
            return np.ones(x.shape[1], dtype=bool)
        return select_nonzero_columns(x, as_bool=True)

    @staticmethod
    def mask_cols_nonnan(
        x: Union[np.ndarray, pd.DataFrame],
        non_nan: Optional[bool] = True
    ):
        """Retain non-nan columns."""
        # When non_nan is False or not set, not necessary to compute
        if non_nan is None or non_nan == False:
            return np.ones(x.shape[1], dtype=bool)
        return select_nonnan_columns(x, non_nan=non_nan, as_bool=True)

    @staticmethod
    def mask_categories(
        dataframe: pd.DataFrame,
        categories: Optional[Dict[str, Sequence]],
        permissive: bool = False
    ) -> np.ndarray:
        """Retain specific categorical column values (by union)."""
        # When categories is None, no need to filter
        if categories is None:
            return np.ones(len(dataframe), dtype=bool)
        # Otherwise, we will 
        permissive = permissive if isinstance(permissive, bool) else False
        # Select items belonging to a category
        categories_mask = np.ones(len(dataframe), dtype=bool)
        for col, selected_categories in categories.items():
            # Check if column from categories is in dataframe
            if not (col in dataframe.columns):
                if not permissive:
                    raise KeyError(f"Column={col} doesn't exist in data")
                continue
            # Update the mask of selected categories
            cmask = select_categories(
                dataframe[col], selected_categories, as_bool=True
            )
            categories_mask = categories_mask & cmask
        return categories_mask


def select_cols(
    cols: Union[List, np.ndarray],
    n_cols: Union[int, List, np.ndarray],
    as_bool: bool = False
) -> np.ndarray:
    """Select the specified columns in [0, n_columns[.

    cols: Sequence of int or Sequence of bool
        Specified column indices of length <= {n_cols}, or a boolean array of
        same length as {n_cols}
    n_cols: int
        Total number of columns in the original data
    as_bool: bool
        When set to True, we return a boolean array for which samples
        fulfilling the condition are set to True

    Returns: np.ndarray
        A 1D array indicating the selected rows indices, or a boolean array
        if {as_bool} is True

    """
    array = select_n_features(n_cols, as_bool=as_bool)
    if as_bool:
        b_array = ~array  # negate to have only False
        b_array[cols] = True
        return b_array
    return array[cols]


def select_colnames(
    colnames: Sequence[str],
    selected_colnames: Union[str, List[str], np.ndarray],
    as_bool: bool = False
) -> np.ndarray:
    """Select the specified columns by name.

    colnames: Sequence[str]
        Column names in the original data, there should be no duplicates
    selected_colnames: str or Sequence of str
        Selected column names
    as_bool: bool
        When set to True, we return a boolean array for which samples
        fulfilling the condition are set to True

    Returns: np.ndarray
        A 1D array indicating the selected cols indices, or a boolean array
        if {as_bool} is True

    """
    b_array = None
    index_array = None
    # Check instances
    if isinstance(colnames, pd.Series):
        colnames = colnames.to_numpy()
    if isinstance(selected_colnames, str):
        selected_colnames = [selected_colnames]
    # Check if no duplicates are found in colnames
    _, c = np.unique(colnames, return_counts=True)
    if any(c > 1):
        raise ValueError("Duplicates found in {colnames}.")
    # Check if no duplicates are found in selected_colnames
    _, c = np.unique(selected_colnames, return_counts=True)
    if any(c > 1):
        raise ValueError("Duplicates found in {selected_colnames}.")
    # Check selection integrity
    if len(set(selected_colnames) - set(colnames)) > 0:
        raise ValueError(
            "Unknown category in {selected_colnames} and not in "
            "{colnames}."
        )
    b_array = np.isin(colnames, selected_colnames)
    if not as_bool:
        index_array = np.arange(len(colnames))[b_array]
    # Return boolean array
    if as_bool:
        return b_array
    # Return selected feature indices
    return index_array


def select_categories(
    feature_category: Union[List[str], pd.Series, np.ndarray],
    selected_category: Union[int, str, Sequence[str], bool],
    as_bool: bool = False
) -> np.ndarray:
    """Select the feature belonging to a category.

    feature_category: list, pd.Series or numpy array of str
        Features category list of shape (,n_features)
    selected_category: str or Sequence of str
        Categories in {feature_category} we want to keep
    as_bool: bool
        When set to True, we return a boolean array for which features
        fulfilling the condition are set to True

    Returns: numpy.ndarray
        An array containing the indices of kept columns, for those found in
        {selected_category}

    Raise ValueError:
        When duplicates are found in {selected_category}, and when category
        in {selected_category} are never seen in {feature_category}

    """
    b_array = None
    index_array = None
    # Check instances
    if isinstance(feature_category, pd.Series):
        feature_category = feature_category.to_numpy()
    if isinstance(selected_category, (str, int, bool)):
        selected_category = [selected_category]
    # Case when the category is a bool
    if isinstance(selected_category[0], bool):
        feature_category = np.asarray(feature_category)
        if feature_category.dtype == bool:
            b_array = feature_category
        else:
            b_array = (feature_category == feature_category)
        if not selected_category:
            b_array = ~b_array
    # 
    else:
        # Check for duplicates
        _, c = np.unique(selected_category, return_counts=True)
        if any(c > 1):
            raise ValueError("Duplicates found in {selected_category}.")
        # Check selection integrity
        if len(set(selected_category) - set(feature_category)) > 0:
            raise ValueError(
                "Unknown category in {selected_category} and not in "
                "{feature_category}."
            )
        b_array = np.isin(feature_category, selected_category)
    if not as_bool:
        index_array = np.arange(len(feature_category))[b_array]
    # Return boolean array
    if as_bool:
        return b_array
    # Return selected feature indices
    return index_array


def select_nonzero_rows(
    x: Union[np.ndarray, pd.DataFrame],
    as_bool: bool = False
) -> np.ndarray:
    """Select n non-zero sample indices in [0, n_samples[.

    x: numpy.ndarray or pd.DataFrame
        2D Array of shape (n_samples, n_features)
    as_bool: bool
        When set to True, we return a boolean array for which features
        fulfilling the condition are set to True

    Returns: numpy.ndarray
        An array containing the indices of non-zero rows, or a boolean
        array indicating features fulfilling the condition

    Raise TypeError, ValueError:
        When x is not a supported data type, or when it is not a 2D array

    """
    b_array = None
    indices_array = None
    # Check types
    if not isinstance(x, (np.ndarray, pd.DataFrame)):
        raise TypeError("Non-supported x type.") 
    # Convert to numpy
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    # A numpy array
    if x.ndim == 2:
        b_array = ~np.all(x == 0, axis=1)
        if not as_bool:
            indices_array = np.arange(x.shape[0])
    else:
        raise ValueError("x.ndim should be equal to 2 (2D array)")
    # Return boolean array
    if as_bool:
        return b_array
    # Return selected feature indices
    return indices_array[b_array]


def select_nonzero_columns(
    x: Union[np.ndarray, pd.DataFrame],
    as_bool: bool = False
) -> np.ndarray:
    """Select n non-zero feature indices in [0, n_features[.

    x: numpy.ndarray or pd.DataFrame
        Array of shape (, n_features) or (n_samples, n_features)
    as_bool: bool
        When set to True, we return a boolean array for which features
        fulfilling the condition are set to True

    Returns: numpy.ndarray
        An array containing the indices of non-zero columns, or a boolean
        array indicating features fulfilling the condition

    Raise TypeError, ValueError:
        When x is not a supported data type, or when it is not a 2D array

    """
    b_array = None
    indices_array = None
    # Check types
    if not isinstance(x, (np.ndarray, pd.DataFrame)):
        raise TypeError("Non-supported x type.") 
    # Convert to numpy
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()
    # A numpy array
    if x.ndim == 2:
        b_array = ~np.all(x == 0, axis=0)
        if not as_bool:
            indices_array = np.arange(x.shape[1])
    else:
        raise ValueError("x.ndim should be equal to 2 (2D array)")
    # Return boolean array
    if as_bool:
        return b_array
    # Return selected feature indices
    return indices_array[b_array]


def select_nonnan_rows(
    x: Union[np.ndarray, pd.DataFrame],
    non_nan: Union[Literal[True], Sequence[str]] = True,
    as_bool: bool = False
) -> np.ndarray:
    """Select n non-nan sample indices in [0, n_samples[.
    
    x: numpy.ndarray or pd.DataFrame
        Array of shape (n_samples, n_features)
    non_nan: True or Sequence of str
        Either (True) indicating that we want to filter nan rows, or a list
        of columns indicating that we filter nan only present in those
    as_bool: bool
        When set to True, we return a boolean array for which features
        fulfilling the condition are set to True

    Returns: numpy.ndarray
        An array containing the indices of non-zero columns, or a boolean
        array indicating features fulfilling the condition

    """
    b_array = None
    indices_array = None
    # Check types
    if not isinstance(x, (np.ndarray, pd.DataFrame)):
        raise TypeError("Non-supported x type.") 
    if non_nan == True:
        # A pandas dataframe
        if isinstance(x, pd.DataFrame):
            b_array = ~(np.any(x.isnull().to_numpy(), axis=1))
        # A numpy array
        elif x.ndim == 2:
            b_array = ~(np.any(np.isnan(x), axis=1))
        else:
            raise ValueError("x.ndim should be equal to 2 (2D array)")
        if not as_bool:
            indices_array = np.arange(x.shape[0])
    # We consider it as a list of column names, x should be a pd.DataFrame
    else:
        if not isinstance(x, pd.DataFrame):
            raise ValueError(
                "x should be a pandas.DataFrame when non_nan is a list of str"
            )
        b_array = np.ones(x.shape[0], dtype=bool)
        for colname in non_nan:
            if not (colname in x.columns):
                raise ValueError(f"{colname=} not found in {x.columns=}")
            b_array = b_array & ~np.isnan(x[colname])
    # Return boolean array
    if as_bool:
        return b_array
    # Return selected feature indices
    return indices_array[b_array]

def select_nonnan_columns(
    x: Union[np.ndarray, pd.DataFrame],
    as_bool: bool = False
) -> np.ndarray:
    """Select n non-nan feature indices in [0, n_features[.

    x: numpy.ndarray or pd.DataFrame
        2D Array of shape (n_samples, n_features)
    as_bool: bool
        When set to True, we return a boolean array for which features
        fulfilling the condition are set to True

    Returns: numpy.ndarray
        An array containing the indices of non-zero columns, or a boolean
        array indicating features fulfilling the condition

    Raise TypeError, ValueError:
        When x is not a supported data type, or when it is not a 2D array

    """
    b_array = None
    indices_array = None
    # Check types
    if not isinstance(x, (np.ndarray, pd.DataFrame)):
        raise TypeError("Non-supported x type.") 
    # A pandas dataframe
    if isinstance(x, pd.DataFrame):
        b_array = ~(np.any(x.isnull().to_numpy(), axis=0))
    # A numpy array
    elif x.ndim == 2:
        b_array = ~(np.any(np.isnan(x), axis=0))
    else:
        raise ValueError("x.ndim should be equal to 2 (2D array)")
    if not as_bool:
        indices_array = np.arange(x.shape[1])
    # Return boolean array
    if as_bool:
        return b_array
    # Return selected feature indices
    return indices_array[b_array]


def select_n_features(
    n_features: Union[int, List, np.ndarray],
    max_n: Optional[int] = None,
    as_bool: bool = False,
    seed: Optional[Any] = None
) -> np.ndarray:
    """Select n non-redundant feature indices in [0, n_features].

    n_features: int, list, np.ndarray
        The number of features, or an array on which to attribute
        n_features=len(array) with array a 1D array of shape (, n_features)
    max_n: int, optional
        The maximum number of feature indice to be selected, when None
        max_n==n_features. When it is set and n_features>max_n then we
        randomly select {max_n_features} values in an array containing
        every indice from 0 to n_features - 1
    as_bool: bool
        When set to True, we return a boolean array for which samples
        fulfilling the condition are set to True
    seed: int or compatible type, optional
        Seed for reproducibility

    Returns: numpy.ndarray
        An array of size {max_n}, it contains the chosen feature indices such
        that every element are in [0, n_features], or a boolean array in case
        {as_bool} is True. When n_features is a tensor, with a specified
        device then the return value is a tensor with same device

    raise ValueError:
        When the number of dimension of the array exceed 1, when {n_features}
        is a list, a numpy array

    """
    # When an array is provided, len of array is the number of features
    if isinstance(n_features, (list, np.ndarray)):
        # Check dim
        if isinstance(n_features, np.ndarray):
            if len(n_features.shape) != 1:
                raise ValueError(
                    "Array should contain only one dimension of "
                    "shape (, n_features)"
                )
        n_features = len(n_features)
    # Main functionality
    max_n_features = n_features if max_n is None else max_n
    b_array = np.zeros(n_features, dtype=bool)
    indices_array = None
    # All features are selected when n_features <= max_n_features
    if n_features <= max_n_features:
        b_array[:] = True
        if not as_bool:
            indices_array = np.arange(n_features)
    # Otherwise a subset of max_n_features is selected across n_features
    elif n_features > max_n_features:
        indices_array = np.random.default_rng(seed=seed).permutation(
            n_features
        )[:max_n_features]
        b_array[indices_array] = True
    # Return boolean array
    if as_bool:
        return b_array
    # Return selected feature indices
    return indices_array


if __name__ == "__main__":
    pass
