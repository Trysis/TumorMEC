import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
import sklearn.inspection as inspection
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection

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
    rand_search = model_selection.RandomizedSearchCV(
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
        class_weight=class_weight
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
