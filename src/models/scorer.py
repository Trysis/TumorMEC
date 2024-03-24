from sklearn import metrics
from sklearn.metrics import make_scorer

__author__ = "Roude JEAN MARIE"
__email__ = "roude.bioinfo@gmail.com"


def generic_score_fn(fn, key=None, to_scorer=False, **kwargs):
    """Utilitary function to provide the specified scorer with
    compatibility for the sklearn {scoring} functions arguments

    fn: callable
        A score function from sklearn.metrics (or compatible)

    key: str or None, default=None
        If specified, we return a dictionnary with
        key the specified {key}

    to_scorer: bool
        Should we return a wrapped scorer function compatible
        with sklearn estimators allowing a {scoring} parameters

    **kwargs:
        Additional key word argument provided to the scorer,
        used to specify default parameters such as coefficient

    Returns: callable
        A callable function providing the score from sklearn.metrics
        or a scorer wrapping the callable obtained
        with sklearn.metrics.make_scorer

    """
    fn = make_scorer(fn, **kwargs) if to_scorer else fn
    to_return = fn if key is None else {key: fn}
    return to_return


def accuracy_score(key=None, to_scorer=False, **kwargs):
    fn = metrics.accuracy_score
    return generic_score_fn(fn=fn, key=key, to_scorer=to_scorer, **kwargs)


def balanced_accuracy_score(key=None, to_scorer=False, **kwargs):
    fn = metrics.balanced_accuracy_score
    return generic_score_fn(fn=fn, key=key, to_scorer=to_scorer, **kwargs)


def precision_score(key=None, to_scorer=False, **kwargs):
    fn = metrics.precision_score
    return generic_score_fn(fn=fn, key=key, to_scorer=to_scorer, **kwargs)


def recall_score(key=None, to_scorer=False, **kwargs):
    fn = metrics.recall_score
    return generic_score_fn(fn=fn, key=key, to_scorer=to_scorer, **kwargs)


def roc_auc_score(key=None, to_scorer=False, **kwargs):
    fn = metrics.roc_auc_score
    return generic_score_fn(fn=fn, key=key, to_scorer=to_scorer, **kwargs)


def matthews_corrcoef(key=None, to_scorer=False, **kwargs):
    fn = metrics.matthews_corrcoef
    return generic_score_fn(fn=fn, key=key, to_scorer=to_scorer, **kwargs)


def f1_score(key=None, to_scorer=False, **kwargs):
    fn = metrics.f1_score
    return generic_score_fn(fn=fn, key=key, to_scorer=to_scorer, **kwargs)


# Global variable
SCORING = {
    "accuracy": accuracy_score(to_scorer=True),
    "balanced_accuracy": balanced_accuracy_score(to_scorer=True),
    "precision": precision_score(to_scorer=True),
    "recall": recall_score(to_scorer=True),
    "auc": roc_auc_score(to_scorer=True),
    "mcc": matthews_corrcoef(to_scorer=True),
    "f1": f1_score(to_scorer=True),
}

if __name__ == "__main__":
    pass
