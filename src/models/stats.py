"""Scripts containing function computing statistics."""
import operator

import scipy

__author__ = "Roude JEAN MARIE"
__email__ = "roude.bioinfo@gmail.com"


def get_pmf(n_trials, probability=0.5):
    """Binomial distribution list of probability for an
    event of {n_trials} trials from a probability mass function

    n_trials: int
        Number of trial

    probability: float
        Probability of succes comprised in [0;1]

    Returns: list -> list from i to {n_trials}
        Probability distribution list of the chance of
        succes for each event {i}

    """
    pmf_list = [
        scipy.stats.binom.pmf(k=k, n=n_trials, p=probability)
        for k in range(n_trials+1)
    ]
    return pmf_list


def get_tail_pmf(pmf_list, alpha=0.05):
    """Returns the index of the element forming the tail
    from 0 to the index according to a probability distribution
    of a symetric shaped distribution such as a bell distribution

    pmf_list: list, tuple or iterable
        List containing the probability for an outcome
        i in a list of n different outcome
    
    alpha: float
        float from 0 to 1 used to define the
        cumulative probability region summing to
        this treshold value, such that each probability
        are additionned up to the treshold value
    
    Returns: int
        index of the last value in the probability mass
        list of values summing up to more or equal than
        the treshold

    """
    cumul_prob = 0
    for i, prob in enumerate(pmf_list):
        cumul_prob += prob
        if cumul_prob >= alpha:
            return i

    return -1


def get_boundaries(treshold=-1, n_trials=-1):
    """Returns the left, middle and right boundary from a specified
    index treshold and the number of trial for symetric shaped
    distribution such as a bell distribution
    
    treshold: int
        Treshold index value

    n_trials: int
        Number of trials

    Returns: tuple -> tuple(tuple(), tuple(), tuple())
        An array of value containing at each index the
        left, middle and right boundary associated with
        the treshold and the number of trials

    """
    if treshold == -1 or n_trials == -1:
        return None, None, None

    if n_trials/2 < treshold:
        return None, None, None

    left = (0, treshold)
    middle = (treshold, n_trials - treshold)
    right = (n_trials - treshold, 1)
    return left, middle, right


def in_boundary(
    value, boundary,
    left_inclusion=True,
    right_inclusion=False
):
    """Check if the value provided is in the specified boundary
    
    value: scalar, such as a float or int
        A value to check if in {boundary}
    
    boundary: list of len==2
        List containing two values associated with
        the min and max boundary

    left_inclusion: bool
        Do we include the left boundary ?

    right_inclusion: bool
        Do we include the right boundary ?

    Returns: bool
        A boolean value that is true if {value} in
        boundary, with the specified parameters.

    """
    op_left, op_right = None, None
    # include left boundary: [, or exclude (,
    op_left = operator.ge if left_inclusion else operator.gt

    # include right boundary: ,] or exclude ,)
    op_right = operator.le if right_inclusion else operator.lt

    to_check = (
        op_left(value, boundary[0]) &
        op_right(value, boundary[-1])
    )
    return to_check


if __name__ == "__main__":
    pass
