import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestClassifier
import sklearn.inspection as inspection

import sklearn.metrics as metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import cross_validate


SEED = 42

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
        groups=None, fold=None,
        scoring=(
            precision_recall_curve,
            roc_curve,
            class_likelihood_ratios,
        ),
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

    fold: int, default=5
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
    scores = cross_validate(
        estimator=estimator,
        X=x,
        y=y,
        groups=groups,
        scoring=scoring,
        cv=fold,
        return_train_score=return_train_score,
        return_estimator=return_estimator,
        return_indices=return_indices,
        **kwargs
    )

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


def forest_importance(rf_model):
    """"""
    importances_mean = rf_model.feature_importances_
    importances_std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)
    return importances_mean, importances_std


def forest_permutation_importance(
    rf_model, x, y, n_repeats=10, random_state=42, n_jobs=2
):
    """"""
    result = inspection.permutation_importance(
        rf, x, y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
    )
    return result

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


if __name__ == "_main__":
    pass
