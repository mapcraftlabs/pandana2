def no_decay(max_weight: float):
    """
    Network aggregations with no decay.  Values will be filtered out where the shortest
        path weight is greater than max_weight.  No decay means a value at max_weight
        will be weighted equally to a value at the origin node.
    :param max_weight: Values beyond max_weight (sum of the edge weight in network
        distance) will not be considered
    :return: A value for the given origin node
    """
    return lambda weights: weights < max_weight


def linear_decay(max_weight: float):
    """
    Network aggregations with linear decay.  Values will be filtered out where the shortest
        path weight is greater than max_weight.  Linear decay means a value at max_weight will
        be weighted as zero while a value at the origin node is weighted at 1 and a weight
        halfway to max_weight will be weighted as 0.5.
    :param max_weight: Values beyond max_weight (sum of weight_col in network distance)
        will not be considered
    :return: A value for the given origin node
    """
    return lambda weights: ((max_weight - weights).clip(lower=0) / max_weight)
