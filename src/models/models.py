import time

import numpy as np

import scipy
from sklearn.utils import Bunch
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
import sklearn.inspection as inspection

import sklearn.metrics as metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GroupShuffleSplit

from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold

# Local module
from . import stats
from . import scorer

__author__ = "Roude JEAN MARIE"
__email__ = "roude.bioinfo@gmail.com"

SEED = 42
SCORING = {
    "accuracy": scorer.accuracy_score(to_scorer=True),
    "balanced_accuracy": scorer.balanced_accuracy_score(to_scorer=True),
    "precision": scorer.precision_score(to_scorer=True),
    "recall": scorer.recall_score(to_scorer=True),
    "auc": scorer.roc_auc_score(to_scorer=True),
    "mcc": scorer.matthews_corrcoef(to_scorer=True),
    "f1": scorer.f1_score(to_scorer=True),
}
ESTIMATOR = RandomForestClassifier
# TODO: Youden Index, Precision Recall Curve

def time_fn(lambda_fn):
    """Time the specified function
    
    lambda_fn: funct
        A lambda function to be called, with the
        pre-specified argument provided

    Returns: sklearn.utils.Bunch
        A dictionnary containing the result of the
        lambda function and the elapsed time

    """
    start_time = time.time()
    result = lambda_fn()
    elapsed_time = time.time() - start_time
    return Bunch(result=result, time=elapsed_time)


def split_xy(df, x_columns, y_columns, groups=None):
    """Returns the set of predictor (x) and the associated
    target (y) from the defined columns without na values

    dataframe: pandas.Dataframe
        A pandas dataframe containing the descriptors
        and the target columns

    x_columns: str or list(str)
        A single column name, or a set of columns
        defined in df

    y_columns: str, list(str)
        A single target column name or a set of column
        defining the selected target columns(s)

    groups: str, list(str)
        Used column to define rows from the same sample,
        it can be a column containing patient identifier
        so that we maintain a patient in the same split
        for future splitting,
        this column won't be in {x} or {y}

    if groups is None:
        Returns: numpy.ndarray, numpy.ndarray
            Respectively the {x} and {y} array
    else:
        Returns: numpy.ndarray, numpy.ndarray, numpy.ndarray
            The {x}, {y} and {groups} array

    """
    if None in (x_columns, y_columns) or df is None:
        raise Exception("Args should not be equals to None")
    if isinstance(x_columns, str):
        x_columns = [x_columns]
    if isinstance(y_columns, str):
        y_columns = [y_columns]
    # Columns verification, if specified columns in {df}
    df_column_set = set(df.columns)
    x_column_set, y_column_set = set(x_columns), set(y_columns)
    groups_set = set(groups) if groups is not None else None
    if not x_column_set.issubset(df_column_set):
        raise Exception(
            "{x_columns} is not a subset of {df.columns}\n"
            f"\t{x_columns = }\n\tdiff={x_column_set - df_column_set}"
        )
    if not set(y_columns).issubset(df_column_set):
        raise Exception(
            "{y_columns} is not a subset of {df.columns}\n"
            f"\t{y_columns = }\n\tdiff={y_column_set - df_column_set}"
        )
    if groups_set is not None and not set(groups).issubset(df_column_set):
        raise Exception(
            "{groups} is not a subset of {df.columns}\n"
            f"\t{groups = }\n\tdiff={groups_set - df_column_set}"
        )
    # Defines the main dataframe without na
    xy = df[x_columns + y_columns].dropna()
    x = xy[x_columns].values  # features
    y = xy[y_columns].values  # target
    if groups is not None:
        if isinstance(groups, str): groups = [groups]
        if len(groups) != 0:
            groups = df.groupby(groups).ngroup().values
            return x, y, groups
        else:
            print("len{groups}=0, only {x} and {y} have been returned")

    return x, y, None


def split_data(
        x, y, groups=None,
        n_splits=1, test_size=0.2,
        stratify=True, seed=SEED
):
    """Create a split of the dataset

    x: array-like of shape (n_samples, n_features)
        data features to fit

    y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        target variable to predict

    groups: array-like of shape (n_samples,), default=None
        group labels for the samples, used
        to split data while conserving a group in
        one set only (train or test)

    n_split: int, default=1
        Number of repeated split of the data

    test_size: float or int, default=0.2
        If float, it correspond to the proportion
        of the dataset to include in the test split.
        If int, represent the absolute number of test
        samples.

    stratify: bool, default=True
        Should the split be stratified ? Such
        that the percentage of the target class
        is the same across the sets (unavailable
        with groups!=None)

    seed: int, default=models.SEED
        Seed to control the randomness and
        have reproductible results

    if n_split == 1:
        if groups is not None:
            Returns: tuple -> tuple(ndarray) * 6
                A tuple containing the list of train and test
                split of inputs, represented as tuple(x_train,
                x_test, y_train, y_test, groups_train, groups_test)
        else:
            x_train, x_test, y_train, y_test

    elif n_split > 1:
        Yiels: tuple -> tuple(ndarray, ndarray)
            The training and test indices represented
            as tuple(train_index, test_index)

    """
    # Return a generator if number of split > 1
    return_generator = n_splits > 1
    data_args = {"X": x, "y":y, "groups": groups}
    split_args = {
        "n_splits": n_splits,
        "test_size": test_size,
        "train_size": None,
        "random_state": seed
    }
    # If the user defined groups for the variable
    if groups is not None:
        gss = GroupShuffleSplit(**split_args)
        gss_gen = gss.split(**data_args)
        if not return_generator:
            train_index, test_index = next(gss_gen)
            return (
                x[train_index], x[test_index],
                y[train_index], y[test_index],
                groups[train_index], groups[test_index]
            )
        else:
            return gss_gen
    # Else case
    else:
        sss = (
            StratifiedShuffleSplit(**split_args) if stratify
            else ShuffleSplit(**split_args)
        )
        sss_gen = sss.split(**data_args)
        if not return_generator:
            train_index, test_index = next(sss_gen)
            return (
                x[train_index], x[test_index],
                y[train_index], y[test_index],
                None, None
            )
        else:
            return sss_gen


def random_forest_search(
    x, y, groups=None, n_split=5,
    stratify=True, seed=SEED, verbose=0,
    scoring=SCORING, n_iter=5, refit="f1", n_jobs=None,
    class_weight="balanced", return_train_score=False,
    cv_generator=None, random_state=SEED,
    param_criterion=["entropy",],
    param_n_estimators=[20, 30, 40, 60],
    param_max_features=["sqrt"],
    param_max_depths=[2, 4, 8, 10, 15, 20],
    param_min_s_split=[2, 4, 16, 32],
    param_min_s_leaf=[1, 5],
    param_bootstrap=[False, True],
):
    """Perform a random search on random forest classifier algorithm

    x: array-like of shape (n_samples, n_features)
        Features

    y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target(s)

    groups: array-like of shape (n_samples,), default=None
        group labels for the samples, should be
        used in conjunction with a Group cv instance
        (e.g., sklearn.model_selection.GroupKFold,
        sklearn.model_selection.StratifiedGroupKFold, ...)

    class_weight: str, default=None
        Associated weights to the target, can be
        "balanced", "balanced_subsample", or None
        See sklearn.ensemble.RandomForestClassifier
        class_weight argument.

    n_split: int, default=5
        Number of repeated split of the data
    
    stratify: bool, default=True
        Should the different split across the
        folds be stratified ? Such that we
        attempt to preserve the percentage
        of the target class across the sets

    seed: int, default=models.SEED
        Seed to control the randomness and
        have reproductible results, control splitting
        and randomsearch sampling

    verbose: int, default=0
        Should we output log output ? verbose can
        equal 1, 2 or 3 for different level of
        verbosity

    scoring: str, callable, list, tuple, or dict
        see the {scoring} argument from
        sklearn.model_selection.cross_validate
        documentation

    n_iter: int, default=5
        Number of parameter settings that are sampled.

    refit: bool, str, or callable, default=True
        Refit the model using the best found parameters,
        for multiple metric evaluation, this needs to be
        a {str} specifying the scorer that will be used.
        See sklearn.model_selection.RandomizedSearchCV
        documentation.

    n_jobs: int, default=None
        Number of jobs to run in parallel for the search
        algorithm. See 'sklearn.model_selection.
        RandomizedSearchCV'

    class_weight: str, dict, list(dict) or None, default="balanced"
        Weights associated with each classes, if not given each
        class is supposed to have weight one.
        -> RandomForestClassifier inner parameters

    random_state: intn RandomState instance or None, default=models.SEED
        Controls the randomness of the random forest algorithm if
        specified such as bootstrapping and best_split.
        See sklearn.ensemble.RandomForestClassifier
        -> RandomForestClassifier inner parameters

    param_{str}: list
        List corresponding to the distribution or
        a set of parameters to try for the corresponding
        {str} parameter name from the estimator object.
        See sklearn.ensemble.RandomForestClassifier
        documentation.

    Returns: object
        Instance of the fitted estimator having
        multiple attributes specified in
        sklearn.model_selection.RandomizedSearchCV
        such as, cv_results_, best_estimator_,
        best_score_, best_params_, best_index_,
        scorer_ and other.

    """
    # Search parameters
    param_search_cv = {
        'criterion': param_criterion,
        'n_estimators': param_n_estimators,
        'max_features': param_max_features,
        'max_depth': param_max_depths,
        'min_samples_split': param_min_s_split,
        'min_samples_leaf': param_min_s_leaf,
        'bootstrap': param_bootstrap,
    }

    # Cross-validation object generator
    cv_generator = cv_object(
        n_split=n_split, groups=groups, stratify=stratify, seed=seed
    ) if cv_generator is None else cv_generator

    # Random search to find best hyperparameters
    random_search = RandomizedSearchCV(
        estimator=ESTIMATOR(class_weight=class_weight, random_state=random_state),
        param_distributions=param_search_cv, 
        n_iter=n_iter, scoring=scoring, cv=cv_generator,
        refit=refit, return_train_score=return_train_score,
        n_jobs=n_jobs, verbose=verbose,
        random_state=seed
    )

    # Fit the random search object
    if groups is not None:
        random_search.fit(x, y, groups=groups)
    else:
        random_search.fit(x, y)

    return random_search


def cross_validation(
        estimator, x, y,
        groups=None, n_split=5,
        scoring=SCORING,
        seed=SEED, stratify=True,
        return_train_score=False,
        return_estimator=False,
        return_indices=False,
        **kwargs
):
    """Perform a cross-validation on the data

    estimator: object
        estimator object implementing fit

    x: array-like of shape (n_samples, n_features)
        data features to fit

    y: array-like of shape (n_samples,) or (n_samples, n_outputs)
        target variable to predict

    groups: array-like of shape (n_samples,), default=None
        group labels for the samples, should be
        used in conjunction with a Group cv instance
        (e.g., sklearn.model_selection.GroupKFold,
        sklearn.model_selection.StratifiedGroupKFold, ...)

    n_split: int, default=5
        Number of repeated split of the data

    scoring: str, callable, list, tuple, or dict
        see the {scoring} argument from
        sklearn.model_selection.cross_validate
        documentation

    seed: int, default=models.SEED
        Seed to control the randomness and
        have reproductible results

    stratify: bool, default=True
        Should the different split across the
        folds be stratified ? Such that we
        attempt to preserve the percentage
        of the target class across the sets

    return_train_score: bool, default=False
        Should we compute and return the train
        score for each split ?
 
    return_estimator: bool, default=False
        Should we return the fitted estimators
        for each split ?

    return_indices: bool, default=False
        Should we return the indices for each
        split ?

    kwargs: dict
        supplementary key word argument

    Returns: dict
        Dictionnary containing the different
        score computed on the different fold,
        also contain fitting time, scoring time.
        train_scores, estimator and indices are
        provided if the user set the argument to
        True.
        See sklearn.model_selection.cross_validate
        documentation

    """
    cv_generator = cv_object(
        groups=groups, n_split=n_split, stratify=stratify, seed=seed
    )

    scores = cross_validate(
        estimator=estimator,
        X=x,
        y=y,
        groups=groups,
        scoring=scoring,
        cv=cv_generator,
        return_train_score=return_train_score,
        return_estimator=return_estimator,
        return_indices=return_indices,
        **kwargs
    )

    return scores 


def cv_object(groups=None, n_split=None, stratify=None, seed=SEED):
    """Returns a cross-validation sklearn object for cross-validation
    depending on the user input. See sklearn.model_selection.KFold,
    StratifiedKFold, GroupKFold and StratifiedGroupKFold

    groups: array-like of shape (n_samples,), default=None
        group labels for the samples, used
        to split data while conserving a group in
        one set only (train or test). If used, then
        the cross-validation object takes into account
        groups (GroupKFold & StratifiedGroupKFold)

    n_split: int, default=None
        Number of repeated split of the data,
        must at least be equal to 2

    stratify: bool, default=True
        Should the split be stratified ? Such
        that the percentage of the target class
        is the same across the sets (unavailable
        with groups!=None)

    Returns: sklearn.model_selection._BaseKFold child class
        The cross-validation object, ihniriting 
        sklearn.model_selection._BaseKFold

    """
    cv_generator = None
    if all(v is None for v in [n_split, groups, stratify]):
        return None

    cv_args = {
        "n_splits": n_split,
        "shuffle": True if seed is not None else False,
        "random_state": seed
    }
    if groups is None:
        cv_generator = (
            KFold(**cv_args) if not stratify
            else StratifiedKFold(**cv_args)
        )
    else:
        cv_generator = (
            GroupKFold(n_splits=cv_args["n_splits"])
            if not stratify
            else StratifiedGroupKFold(**cv_args)
        )

    return cv_generator


def predict_model(x, y, estimator):
    """Predict y with a given model"""
    observed = y.copy()
    predicted = estimator.predict(x)
    return observed, predicted


def predict_kmodel(x, y, estimator, kfolder, return_train=False):
    """"""
    observed_train, predicted_train = np.array([]), np.array([])
    observed_test, predicted_test = np.array([]), np.array([])
    for train_idx, test_idx in kfolder:
        train_x, train_y, test_x, test_y = \
            x[train_idx], y[train_idx], x[test_idx], y[test_idx]
        # Predict the labels of the test set samples
        predicted_test = np.append(predicted_test, estimator.predict(test_x))
        observed_test = np.append(observed_test, test_y)
        if return_train:
            predicted_train = np.append(predicted_train, estimator.predict(train_x))
            observed_train = np.append(observed_train, train_y)
    
    if return_train:
        return observed_train, predicted_train, observed_test, predicted_test
    return observed_test, predicted_test


def scorer_model(estimator, x, y, scorer, y_pred=None, for_pandas=True, **kwargs):
    """ Returns the score associated with the estimator prediction
    on data.

    estimator: 
        A sklearn-like model with predict, fit, (...) functions
 
    x: numpy.ndarray
        input features

    y: numpy.ndarray
        target data

    scorer: callable or dict({key: callable, ...})
        Scorer function(s) to apply between the predicted and
        observed {y} values

    y_pred: numpy.ndarray ,optional
        Optional already predicted y value from {x}

    **kwargs:
        Key word optional argument to provide to scorer(s) functions

    Returns: dict
        Dictionnary containing as key provided scorer names,
        and as value the associated score. It returns {"score": value}
        if only a callable is provided.

    """
    observed, predicted = None, None
    if (y_pred is None):
        observed, predicted = predict_model(x, y, estimator)
    else:
        observed, predicted = y, y_pred
    if (observed is None) or (predicted is None):
        raise Exception("observed or predicted is None")
    elif (observed.shape[0] != predicted.shape[0]):
        raise Exception(f"Shape not the same {observed.shape=}; {predicted.shape=}")
    #        
    returned_score = dict()
    if callable(scorer):
        scorer = {"score": scorer}
    for key, to_call in scorer.items():
        returned_score[key] = to_call(
            observed, predicted, **kwargs
        )
        if for_pandas:
            returned_score[key] = returned_score[key].tolist()
            if hasattr(returned_score[key], "tolist"):
                returned_score[key] = returned_score[key].tolist()
            elif isinstance(returned_score[key], (int, float)):
                returned_score[key] = [returned_score[key]]

    return returned_score


def forest_mdi_importance(rf_estimator, colnames, **kwargs):
    """Retrieve computed mdi importance of RandomForest estimator
    
    rf_estimator: object
        estimator object implementing fit

    colnames: list(str)
        column names used during estimator fitting,
        the order should be kept the same as when fitted

    **kwargs:
        Added for compatibility

    Returns: sklearn.utils.Bunch <- dict like
        A "dictionnary" containing mean importance values
        with std and associated colnames.

    """
    importances_mean = rf_estimator.feature_importances_
    importances_std = np.std(
        [tree.feature_importances_ for tree in rf_estimator.estimators_],
        axis=0
    )

    # Importance values
    result = Bunch(
        importances_mean=importances_mean,
        importances_std=importances_std,
        colnames=colnames
    )
    return result


def forest_permutation_importance(
    estimator, x, y, colnames,
    scoring=None,
    n_repeats=10,
    seed=SEED,
    n_jobs=None,
    **kwargs
):
    """Compute permutation importance an the fitted estimator.

    x: ndarray or DataFrame, of shape (n_samples, n_features)
        Dataframe on which permutation importance is computed

    y: array-like, of shape (n_samples,) or (n_samples, n_classes)
        Target class for supervised learning

    colnames: list(str), of size=n_features
        Associated colnames of {x}

    scoring: str, callable, list, tuple or dict, default=None
        Score used to compute importance,
        Same as sklearn.inspection.permutation_importance

    n_repeats: int, default=10
        Number of times to permute a feature

    seed: int
        Random seed

    n_jobs: int or None, default=None
        Number of jobs to run in parallel

    **kwargs:
        Added for compatibility

    Returns: sklearn.utils.Bunch <- dict like
        A "dictionnary" containing as key {importances_mean},
        {importances_std}, and {colnames} keys

    """
    importances = inspection.permutation_importance(
        estimator=estimator,
        X=x, y=y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=seed,
        n_jobs=n_jobs
    )

    result = Bunch(**importances, colnames=colnames)
    return result


def to_boruta_data(x, colnames, prefix="shadow_"):
    """Generate a boruta dataframe such that we add shadow
    features to the original dataframe. shadow features correspond
    to shuffled features of the original features.

    x: ndarray, of shape (n_samples, n_features)
        Dataframe to add shadow features on

    colnames: list(str), of size=n_features
        Associated column names

    prefix: str, default="shadow_"
        shadow features will be prefixed with {prefix}

    Returns: sklearn.utils.Bunch
        A "dictionnary" containing the keys:
            x_boruta: ndarray of shape (n_samples, n_features*2)
            colnames_boruta: associated column names with shadow columns
            prefix: the specified prefix of shadow columns

    """
    if prefix is None or prefix=="":
        prefix = "shadow_"

    # Append shadow features colnames
    shadow_colnames = list(map(
        lambda x: f"{prefix}{x}", colnames
    ))
    colnames_boruta = colnames + shadow_colnames

    # Add shadow features: shuffled columns
    x_shuffled = np.apply_along_axis(
        func1d=np.random.permutation, axis=0, arr=x
    )
    x_boruta = np.concatenate([x, x_shuffled], axis=1)

    # Return data and associated colnames
    result = Bunch(
        x_boruta=x_boruta,
        colnames_boruta=colnames_boruta,
        prefix=prefix
    )
    return result


def run_boruta(
    estimator, x, y, colnames,
    feature_hit=None, clone_estimator=True
):
    """Perform the boruta algorithm, 
    estimator: object
        A fitted or unfitted estimator that
        will be cloned

    x: numpy.ndarray of shape (n_samples, n_features)
        Features of the data

    y: numpy.ndarray of shape (n_features,)
        Target data

    colnames: list, tuple, or numpy.ndarray
        column name of each features, in the
        same order as provided by x

    feature_hit: dict -> {colname: hit, ...}, default=None
        Dictionnary containing the number of times
        a feature fulfilling the boruta selection
        criterion corresponding to
        Features having an higher score than the most
        important shadow feature

    clone_estimator: bool
        Should we clone a new unfitted estimator
        with the same parameters ? Should be used
        if the estimator has already been fit.
        It uses sklearn.base.clone function

    Returns: sklearn.utils.Bunch <-> ihnerit dict
        object attributes:
            estimator, x, y, colnames, importances,
            highest_shadow_col, feature_hit, boruta_dict

        A dictionnary containing the fitted attribute
        {estimator}, the feature {x}, target {y} data
        and associated column names {colnames}, the
        feature importance named as {importances} of
        the column features and the shadow column features.
        The {feature_hit} attribute, contains the number
        of columns fulfilling the boruta selection criterion.

    """
    if clone_estimator:
        estimator = clone(estimator)

    boruta_dict = to_boruta_data(x=x, colnames=colnames)
    boruta_x = boruta_dict.x_boruta
    boruta_colnames = boruta_dict.colnames_boruta
    boruta_prefix = boruta_dict.prefix

    # Fit model with shadow features
    estimator.fit(boruta_x, y)

    # feature importances
    f_importances = {
        f_name: f_imp for f_name, f_imp in
        zip(boruta_colnames, estimator.feature_importances_)
    }
    # only shadow feature importance
    shadow_f_importances = {
        f_name: f_imp for f_name, f_imp in f_importances.items()
        if boruta_prefix in f_name
    }

    # Most important shadow feature to apply criterion on
    highest_shadow_f = max(shadow_f_importances, key=shadow_f_importances.get)
    if feature_hit is None:
        feature_hit = {f_name: 0 for f_name in colnames}

    for f_name in feature_hit:
        if f_importances[f_name] > shadow_f_importances[highest_shadow_f]:
            feature_hit[f_name] += 1

    result = Bunch(
        estimator=estimator,
        x=x, y=y,
        colnames=colnames,
        importances=f_importances,
        highest_shadow_col=highest_shadow_f,
        feature_hit=feature_hit,
        boruta_dict=boruta_dict
    )

    return result


def select_feature_hit(feature_hit, n_trials, prob=0.5, alpha=0.05):
    """Returns the features found as 'non-important', 'unsure' and 'important'
    based on the number of features hit based on an importance algorithm performing
    features hit such as boruta algorithm.

    feature_hit: dict -> {colname: hit, ...}, default=None
        Dictionnary containing the number of times
        a feature which fulfill boruta selection
        criteria: Features having an higher score
        than the most important shadow feature

    n_trials: int
        Number of trials performed for the selected
        algorithm. It is used for comparison with the
        number of hit for each features.

    prob: float
        float value in [0;1] associated with the probability
        of an event to occur

    alpha: float
        Alpha criterion used to estimate an event as
        non-random

    Returns: sklearn.utils.Bunch <- dict like
        It contains all informations about the features
        found as 'non-important', 'unsure', and 'important'
        in left_hit, middle_hit and right hit keys.
        Threshold are also given.

    """
    # Mass probability for the different outcome to be probable
    probability_mass_l = stats.get_pmf(n_trials=n_trials, probability=prob)
    treshold = stats.get_tail_pmf(pmf_list=probability_mass_l, alpha=alpha)
    # Boundaries f
    left_boundary, middle_boundary, right_boundary = \
        stats.get_boundaries(treshold=treshold, n_trials=n_trials)
    
    left_l, middle_l, right_l = list(), list(), list()
    for f_name, hit in feature_hit.items():
        # left
        if stats.in_boundary(
            value=hit, boundary=left_boundary,
            left_inclusion=True, right_inclusion=False
        ):
            left_l.append(f_name)
        # middle
        elif stats.in_boundary(
            value=hit, boundary=middle_boundary,
            left_inclusion=True, right_inclusion=False
        ):
            middle_l.append(f_name)
        else:
            right_l.append(f_name)

    result = Bunch(
        left_hit=left_l,
        middle_hit=middle_l,
        right_hit=right_l,
        lower_treshold=middle_boundary[0],
        upper_treshold=middle_boundary[1],
        treshold=treshold,
    )

    return result


# TODO : Multiprocessing (n_jobs)
def forest_boruta_importance(estimator, x, y, colnames, n_trials=50, alpha=0.05):
    """Compute boruta importance on the given estimator.

    estimator: object
        A fitted or unfitted estimator that
        will be cloned

    x: numpy.ndarray of shape (n_samples, n_features)
        Features of the data

    y: numpy.ndarray of shape (n_features,)
        Target data

    colnames: list, tuple, or numpy.ndarray
        column name of each features, in the
        same order as provided by x

    n_trials: int, default=50
        Number of trials to perform boruta algorithm, for
        estimating the number of feature hit

    alpha: float
        alpha criterion to consider an event as not "random"

    Returns: sklearn.utils.Bunch <- dict like
        A "dictionnary" containing the important, and non-important
        features from the given model. With other information
        such as the lower and upper treshold defining non-important,
        unsure and important features.

    """
    # Run the boruta criterion {n_trials} time and count hit
    feature_hit = None
    for _ in range(n_trials):
        boruta_i = run_boruta(
            estimator=estimator, x=x, y=y, colnames=colnames,
            feature_hit=feature_hit, clone_estimator=True
        )
        feature_hit = boruta_i.feature_hit

    # Apply stats to find important & non-important features
    selection_summary = select_feature_hit(
        feature_hit=feature_hit,
        n_trials=n_trials, alpha=alpha
    )

    # Final summary result from the features
    result = Bunch(
        important = selection_summary.right_hit,
        non_important = selection_summary.left_hit,
        unsure_important = selection_summary.middle_hit,
        lower_treshold = selection_summary.lower_treshold,
        upper_treshold = selection_summary.upper_treshold,
        treshold = selection_summary.treshold,
        feature_hit=feature_hit,
        n_trials=n_trials,
        alpha=alpha
    )

    return result


if __name__ == "_main__":
    pass
