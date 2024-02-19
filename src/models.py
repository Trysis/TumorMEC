import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scipy
from sklearn.utils import Bunch
from sklearn.base import clone
import sklearn.preprocessing as preprocessing
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
import auxiliary

SEED = 42

# TODO: Youden Index, Precision Recall Curve


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
        Returns: tuple -> tuple(ndarray) * 6
            A tuple containing the list of train and test
            split of inputs, represented as tuple(x_train,
            x_test, y_train, y_test, groups_train, groups_test)

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
                y[train_index], y[test_index]
            )
        else:
            return sss_gen


def cross_validation(
        estimator, x, y,
        groups=None, n_split=5,
        scoring={
            "accuracy": metrics.accuracy_score,
            "balanced_accuracy": metrics.balanced_accuracy_score,
            "precision": metrics.precision_score,
            "recall": metrics.recall_score,
            "auc": metrics.roc_auc_score,
            "mcc": metrics.matthews_corrcoef,
            "f1": metrics.f1_score,
        },
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
        supplementary argument

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


def forest_depth_acc(
    x_train, y_train, x_test, y_test,
    classifier=RandomForestClassifier,
    depths=None, **kwargs
):
    """"""
    max_depths = np.arange(10) + 1 if depths is None else depths
    results = {
        "depths": max_depths,
        "auc_train": [],
        "acc_train": [],
        "mcc_train": [],
        "auc_test": [],
        "acc_test": [],
        "mcc_test": []
    }

    for depth in max_depths:
        # Fit model with specified depth
        clf = classifier(max_depth=depth, **kwargs)
        clf.fit(x_train, y_train)
        # Train
        y_train_pred = clf.predict(x_train)
        fpr_train, tpr_train, _ = metrics.roc_curve(y_train, y_train_pred)
        train_auc = metrics.auc(fpr_train, tpr_train)
        train_accuracy = metrics.accuracy_score(y_train, y_train_pred)
        train_mcc = metrics.matthews_corrcoef(y_train, y_train_pred)
        # Test
        y_test_pred = clf.predict(x_test)
        fpr_test, tpr_test, _ = metrics.roc_curve(y_test, y_test_pred)
        test_auc = metrics.auc(fpr_test, tpr_test)
        test_accuracy = metrics.accuracy_score(y_test, y_test_pred)
        test_mcc = metrics.matthews_corrcoef(y_test, y_test_pred)

        # Append metrics
        results["auc_train"].append(train_auc)
        results["acc_train"].append(train_accuracy)
        results["mcc_train"].append(train_mcc)

        results["auc_test"].append(test_auc)
        results["acc_test"].append(test_accuracy)
        results["mcc_test"].append(test_mcc)

    return results


def forest_mdi_importance(rf_estimator, colnames):
    """"""
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
    n_jobs=None
):
    """"""
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


def to_boruta_data(x, colnames, y=None, prefix="shadow_"):
    """
    """
    # Append shadow features colnames
    shadow_colnames = list(map(
        lambda x: f"{prefix}{x}", colnames
    ))
    boruta_colnames = colnames + shadow_colnames

    # Add shadow features: shuffled columns
    x_shuffled = np.apply_along_axis(
        func1d=np.random.permutation, axis=0, arr=x
    )
    x_boruta = np.concatenate([x, x_shuffled], axis=1)

    # Return data and associated colnames
    result = Bunch(
        x=x,
        y=y,
        colnames=colnames,
        x_boruta=x_boruta,
        colnames_boruta=boruta_colnames,
        prefix=prefix
    )
    return result


def run_boruta(
    estimator, x, y, colnames,
    feature_hit=None, clone_estimator=True
):
    """
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
        a feature which fulfill boruta selection
        criteria: Features having an higher score
        than the most important shadow feature

    clone_estimator: bool
        Should we clone a new unfitted estimator
        with the same parameters ?
        Should be used if the estimator has already
        been fit.
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

    boruta_dict = to_boruta_data(x=x, y=y, colnames=colnames)
    boruta_x = boruta_dict.x_boruta
    boruta_colnames = boruta_dict.boruta_colnames
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

    highest_shadow_f = max(shadow_f_importances, key=shadow_f_importances.get)
    if feature_hit is None:
        feature_hit = {f_name: 0 for f_name in colnames}

    for f_name in feature_hit:
        if f_importances[f_name] > highest_shadow_f:
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


def rf_boruta_importance(estimator, x, y, colnames, n_run=50):
    """
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
        a feature which fulfill boruta selection
        criteria: Features having an higher score
        than the most important shadow feature

    clone_estimator: bool
        Should we clone a new unfitted estimator
        with the same parameters ?
        Should be used if the estimator has already
        been fit.
        It uses sklearn.base.clone function

    Returns:
    """
    feature_hit = None
    for _ in range(n_run):
        boruta_i = run_boruta(
            estimator=estimator, x=x, y=y, colnames=colnames,
            feature_hit=feature_hit, clone_estimator=True
        )
        feature_hit = boruta_i.feature_hit

    # Mass probability for the different outcome to be probable
    prob_mass_fn = auxiliary.get_pmf_list(n_run=n_run, probability=0.5)
    [
        scipy.stats.binom.pmf(k=k, n=n_run, p=0.5)
        for k in range(n_run+1)
    ]


def select_feature_hit(feature_hit, n_run, alpha=0.05):
    # Boundaries f
    left_boundary = (0, treshold)
    middle_boundary = (treshold, n_run - treshold)
    right_boundary = (n_run - treshold, 1)



def random_forest_importance(
        estimator, importance_fn=None, timing=True
):
    """"""
    start_time = time.time()
    importance = importance_fn(estimator, **args)
    elapsed_time = time.time() - start_time

    return Bunch(**importance, "time"=elapsed_time)


# Imported from 
# github.com/AdrianaLecourieux/Image-Analysis-for-spatial-transcriptomic/blob/main/analysis/Random_forest.ipynb
def perform_random_forest(
    x_train, x_test, y_train, y_test, columns,
    n_estimators = (20, 30, 40, 60),
    n_depths = (2, 4, 8, 10, 15, 20),
    class_weight=None, scale=False, seed=None,
    verbose=True
):
    """Perform random forest and exctract feature importances.

    Parameters
    ----------
    x_train, x_test: pandas dataframe
        Train and Test features
    Y_train, Y_test: pandas dataframe
        Train and Test targets

    """
    # Standardization
    if scale:
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    param_dist = {
        'n_estimators': n_estimators,
        'max_depth': n_depths
    }

    # Random search to find best hyperparameters
    rf = RandomForestClassifier()
    rand_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, 
        n_iter=5, cv=5, random_state=seed
    )

    rand_search.fit(x_train, y_train)  # Fit the random search object

    # Best hyperparameters
    print('Best hyperparameters:', rand_search.best_params_) if verbose else None
    max_depth = rand_search.best_params_['max_depth']
    n_estimators = rand_search.best_params_['n_estimators']

    # Model with best parameters
    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=seed
    )
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy) if verbose else None

    # Features importance - MDI
    start_time = time.time()
    print("Computing MDI importances") if verbose else None
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(
        f"Elapsed time to compute mean decrease impurity (MDI) "
        f"importance: {elapsed_time:.3f} seconds\n"
    ) if verbose else None

    # Plot - MDI
    forest_importances = pd.Series(importances, index=columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    # Features importance - permutation
    start_time = time.time()
    print("Computing permutation importances ") if verbose else None
    result = inspection.permutation_importance(
        rf, x_test, y_test, n_repeats=10, random_state=seed, n_jobs=-1
    )
    elapsed_time = time.time() - start_time
    print(
        f"Elapsed time to compute permutation importance: "
        f"{elapsed_time:.3f} seconds\n"
    ) if verbose else None

    # Plot - permutation
    forest_importances = pd.Series(result.importances_mean, index=columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation method")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

    to_return = {
        "model": rf,
        "column": columns,
        "importance_mdi": {"mean": importances, "std": std},
        "importance_permutation": {"mean": result.importances_mean, "std": result.importances_std}
    }

    return to_return


def random_forest_search(
    x_train, y_train, groups=None,
    class_weight="balanced", n_split=5,
    stratify=True, seed=SEED, verbose=0,
    scoring={
            "accuracy": metrics.accuracy_score,
            "balanced_accuracy": metrics.balanced_accuracy_score,
            "precision": metrics.precision_score,
            "recall": metrics.recall_score,
            "auc": metrics.roc_auc_score,
            "mcc": metrics.matthews_corrcoef,
            "f1": metrics.f1_score,
    },
    n_iter=5, refit=True, n_jobs=None,
    param_criterion = ("entropy",),
    param_n_estimators = (20, 30, 40, 60),
    param_max_features = ("sqrt,"),
    param_max_depths = (2, 4, 8, 10, 15, 20),
    param_min_s_split = (2, 4, 16, 32),
    param_min_s_leaf = (1, 5),
    param_bootstrap = (False, True),
    param_random_state = SEED
):
    """Perform a random searchrandom forest and exctract feature importances.

    x_train, x_test: pandas dataframe
        Train and Test features

    y_train, y_test: pandas dataframe
        Train and Test targets

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
        have reproductible results

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
        'max_feature': param_max_features,
        'max_depth': param_max_depths,
        'min_samples_split': param_min_s_split,
        'min_samples_leaf': param_min_s_leaf,
        'bootstrap': param_bootstrap,
        'class_weight': class_weight,
        'random_state': param_random_state
    }

    # Cross-validation object generator
    cv_generator = cv_object(
        n_split=n_split, groups=groups, stratify=stratify, seed=seed
    )

    # Random search to find best hyperparameters
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(),
        param_distributions=param_search_cv, 
        n_iter=n_iter, scoring=scoring, cv=cv_generator,
        refit=refit, n_jobs=n_jobs, verbose=verbose,
        random_state=seed
    )

    # Fit the random search object
    random_search.fit(x_train, y_train)

    return random_search


def x(x_test=None, y_test=None,
    colnames=None, ):

    # Best hyperparameters
    print('Best hyperparameters:', rand_search.best_params_) if verbose else None
    max_depth = rand_search.best_params_['max_depth']
    n_estimators = rand_search.best_params_['n_estimators']

    # Model with best parameters
    rf = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=seed
    )
    rf.fit(x_train, y_train)

    y_pred = rf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy) if verbose else None

    # Features importance - MDI
    start_time = time.time()
    print("Computing MDI importances") if verbose else None
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(
        f"Elapsed time to compute mean decrease impurity (MDI) "
        f"importance: {elapsed_time:.3f} seconds\n"
    ) if verbose else None

    # Plot - MDI
    forest_importances = pd.Series(importances, index=columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    # Features importance - permutation
    start_time = time.time()
    print("Computing permutation importances ") if verbose else None
    result = inspection.permutation_importance(
        rf, x_test, y_test, n_repeats=10, random_state=seed, n_jobs=-1
    )
    elapsed_time = time.time() - start_time
    print(
        f"Elapsed time to compute permutation importance: "
        f"{elapsed_time:.3f} seconds\n"
    ) if verbose else None

    # Plot - permutation
    forest_importances = pd.Series(result.importances_mean, index=columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation method")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

    to_return = {
        "model": rf,
        "column": columns,
        "importance_mdi": {"mean": importances, "std": std},
        "importance_permutation": {"mean": result.importances_mean, "std": result.importances_std}
    }

    return to_return


if __name__ == "_main__":
    pass
