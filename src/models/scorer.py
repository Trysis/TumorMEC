# Internal modules
from typing import Callable, Optional

# External modules
from sklearn import metrics
from sklearn.metrics import make_scorer


__author__ = "Roude JEAN MARIE"
__email__ = "roude.bioinfo@gmail.com"


def generic_score_fn(
    fn: Callable, wrap_scorer: bool = False, key: Optional[str] = None, **kwargs
):
    """Utilitary function to provide the specified scorer with
    compatibility for the sklearn {scoring} functions arguments

    fn: callable
        A score function from sklearn.metrics (or compatible)

    key: str or None, default=None
        If specified, we return a dictionnary with
        key the specified {key}

    wrap_scorer: bool
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
    fn = make_scorer(fn, **kwargs) if wrap_scorer else fn
    to_return = fn if key is None else {key: fn}
    return to_return


def accuracy_score(wrap_scorer: bool = False, key: str = "accuracy", **kwargs):
    fn = metrics.accuracy_score
    return generic_score_fn(fn=fn, key=key, wrap_scorer=wrap_scorer, **kwargs)


def balanced_accuracy_score(wrap_scorer: bool = False, key: str = "balanced_accuracy", **kwargs):
    fn = metrics.balanced_accuracy_score
    return generic_score_fn(fn=fn, key=key, wrap_scorer=wrap_scorer, **kwargs)


def precision_score(wrap_scorer: bool = False, key: str = "precision", **kwargs):
    fn = metrics.precision_score
    return generic_score_fn(fn=fn, key=key, wrap_scorer=wrap_scorer, **kwargs)


def recall_score(wrap_scorer: bool = False, key: str = "recall", **kwargs):
    fn = metrics.recall_score
    return generic_score_fn(fn=fn, key=key, wrap_scorer=wrap_scorer, **kwargs)


def roc_auc_score(wrap_scorer: bool = False, key: str = "auc", **kwargs):
    fn = metrics.roc_auc_score
    return generic_score_fn(fn=fn, key=key, wrap_scorer=wrap_scorer, **kwargs)


def matthews_corrcoef(wrap_scorer: bool = False, key: str = "mcc", **kwargs):
    fn = metrics.matthews_corrcoef
    return generic_score_fn(fn=fn, key=key, wrap_scorer=wrap_scorer, **kwargs)


def f1_score(wrap_scorer: bool = False, key: str = "f1", **kwargs):
    fn = metrics.f1_score
    return generic_score_fn(fn=fn, key=key, wrap_scorer=wrap_scorer, **kwargs)


if __name__ == "__main__":
    pass
