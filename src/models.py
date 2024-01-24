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

    # From depth 1 to n
    for depth in max_depths:
        # Fit model with specified depth
        rf = classifier(max_depth=depth, **kwargs)
        rf.fit(x_train, y_train)  
        # Train
        train_pred = rf.predict(x_train)
        fpr_train, tpr_train, _ = metrics.roc_curve(y_train, train_pred)
        train_auc = metrics.auc(false_positive_rate, true_positive_rate)
        train_accuracy = metrics.accuracy_score(y_train, train_pred)
        # Test
        test_pred = rf.predict(x_test)
        false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_test, test_pred)
        test_roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
        test_accuracy = metrics.accuracy_score(y_test, test_pred)

        results["auc_train"].append(train_roc_auc)
        results["acc_train"].append(train_accuracy)
        results["auc_test"].append(test_roc_auc)
        results["acc_test"].append(test_accuracy)

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

# Imported from https://github.com/AdrianaLecourieux/Image-Analysis-for-spatial-transcriptomic/blob/main/analysis/Random_forest.ipynb
# repository
def perform_random_forest(
    X_train, X_test, y_train, y_test, columns,
    n_estimators = (20, 30, 40, 60),
    n_depths = (2, 4, 8, 10, 15, 20),
    class_weight=None, scale=False
):
    """Perform random forest and exctract feature importances.

    Parameters
    ----------
    X_train, X_test: pandas dataframe
        Train and Test features
    Y_train, Y_test: pandas dataframe
        Train and Test targets

    """
    # Standardization
    if scale:
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    param_dist = {
        'n_estimators': n_estimators,
        'max_depth': n_depths
    }

    # Create a random forest classifier
    rf = RandomForestClassifier()

    # Use random search to find the best hyperparameters
    rand_search = model_selection.RandomizedSearchCV(
        rf, 
        param_distributions = param_dist, 
        n_iter=5, 
        cv=5
    )

    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)
    # Print the best hyperparameters
    print('Best hyperparameters:', rand_search.best_params_)
    max_depth = rand_search.best_params_['max_depth']
    n_estimators = rand_search.best_params_['n_estimators']

    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, class_weight=class_weight)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)

    start_time = time.time()
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(importances, index=columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    # Permutation Importance
    start_time = time.time()
    result = inspection.permutation_importance(
        rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    forest_importances = pd.Series(result.importances_mean, index=columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

    #scores = model_selection.cross_val_score(rf, X_train, y_train, cv=5)

    #print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    to_return = {
        "model": rf,
        "column": columns,
        "importance_mdi": {"mean": importances, "std": std},
        "importance_permutation": {"mean": result.importances_mean, "std": result.importances_std}
    }

    return to_return


if __name__ == "_main__":
    pass
