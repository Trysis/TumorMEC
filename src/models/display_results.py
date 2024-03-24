import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

__author__ = "Roude JEAN MARIE"
__email__ = "roude.bioinfo@gmail.com"


def display_cv_scores(
    cv_scores, figsize=None,
    title="", bottom=-1, top=1, 
    filepath=None, show=False
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    cv_unique = cv_scores["model"].unique()
    for sc in cv_unique:
        cv_score = cv_scores[cv_scores["model"] == sc]
        ax.plot(cv_score["score"], cv_score["mean"], label=f"model {sc}", marker="o")
    ax.set_ylim(bottom, top)
    ax.set_xticklabels(cv_scores["score"], rotation=45, ha='right')
    ax.legend()
    if filepath is not None:
        fig.savefig(filepath)
    if show:
        plt.show()
    return fig, ax


def display_confusion_matrix(
    observed, predicted, labels=None, cmap="Blues",
    normalize=False, figsize=None, title="",
    by_col=True, filepath=None, show=False
):
    """"""
    if labels is None:
        labels = [str(i) for i in np.unique(observed)]
    # Confusion matrix values
    cf_matrix = confusion_matrix(observed, predicted)
    cf_matrix_norm = confusion_matrix(observed, predicted, normalize=normalize)
    # To plot
    fig, ax = (
        plt.subplots(figsize=figsize) if not normalize
        else plt.subplots(1, 2, figsize=figsize) if by_col
        else plt.subplots(2, 1, figsize=figsize)
    )
    ax, ax_norm = ax, None if not normalize else ax[0], ax[1]
    im = ax.imshow(cf_matrix, interpolation='nearest', cmap=plt.get_cmap(cmap))
    im_norm = None
    if normalize:
        im_norm = ax_norm.imshow(cf_matrix_norm, interpolation='nearest', cmap=plt.get_cmap(cmap))
    # x & y labels
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks, labels, rotation=45)
    ax.set_yticks(tick_marks, labels)
    # Format
    fmt = '.2f' if normalize else 'd'
    thresh = cf_matrix.max() / 2.
    thresh_norm = cf_matrix_norm.max() / 2.
    for i, j in itertools.product(range(cf_matrix.shape[0]), range(cf_matrix.shape[1])):
        ax.text(j, i, format(cf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cf_matrix[i, j] > thresh else "black")
        if normalize:
            ax_norm.text(
                j, i, format(cf_matrix_norm[i, j], fmt), horizontalalignment="center",
                color="white" if cf_matrix_norm[i, j] > thresh_norm else "black"
            )

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    if normalize:
        ax_norm.set_xlabel('Predicted label')
        ax_norm.set_ylabel('True label')
        fig.colorbar(im_norm)
    else:
        fig.colorbar(im)
    fig.suptitle(title)
    fig.tight_layout()
    if filepath is not None:
        fig.savefig(filepath)
    if show:
        plt.show()

    return fig, ax

def x(x_test=None, y_test=None,
    colnames=None, ):
    """
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

    return to_return"""

if __name__ == "__main__":
    pass
