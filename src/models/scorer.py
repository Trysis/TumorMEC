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
    
    return fn

def balanced_accuracy_score():
    pass

def precision_score():
    pass
"accuracy": ,
                    "balanced_accuracy": metrics.balanced_accuracy_score,
                    "precision": metrics.precision_score,
                    "recall": metrics.recall_score,
                    "auc": metrics.roc_auc_score,
                    "mcc": metrics.matthews_corrcoef,
                    "f1": metrics.f1_score,