import operator

import scipy


def get_pmf(n_run, probability=0.5):
    """Binomial distribution list of probability 
    for an event of {n_run} trials"""
    prob_mass_fn = [
        scipy.stats.binom.pmf(k=k, n=n_run, p=probability)
        for k in range(n_run+1)
    ]
    return prob_mass_fn


def get_tail_pmf(pmf_list, alpha=0.05):
    """Returns the index of the element forming the tail
    from 0 to the index according to a mass probability list
    of a bell shaped distribution
    
    prob_mass_l: list, tuple or iterable
        List containing the probability for an outcome
        i in a list of n different outcome
    
    treshold: float
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


def get_tail_boundaries(treshold=-1, n_run=-1):
    """Returns the left, middle and right boundary from
    a specified index treshold and the number of trial
    for a bell curve shaped distribution function"""
    if treshold == -1 or n_run == -1:
        return None

    left = (0, treshold)
    middle = (treshold, n_run - treshold)
    right = (n_run - treshold, 1)
    return left, middle, right


def in_boundary(
    value, boundary,
    left_inclusion=True,
    right_inclusion=False
):
    """Check if the value provided is in the specified
    boundary"""
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
