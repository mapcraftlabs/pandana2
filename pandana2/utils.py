import numpy as np


def weighted_median(data, weights):
    """
    Calculates the weighted median of a dataset.  Note that this median returns the
        first element with a weight greater than the target weight.  It does *not*
        calculate the median when there is an even number of elements by averaging
        the two in the middle (it just picks the lesser of the two values).  Perhaps
        this should be fixed, but it seems consistent with the spirit of the weighted
        median here.

    Parameters:
    data (np.array): Array of data values.
    weights (np.array): Array of weights corresponding to the data values.

    Returns:
    float: The weighted median.
    """
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]

    cumulative_weights = np.cumsum(sorted_weights)
    median_index = np.where(cumulative_weights >= np.sum(weights) / 2)[0][0]

    return sorted_data[median_index]


def weighted_std(values, weights):
    """
    Calculates the weighted standard deviation.

    Parameters:
    values (numpy.ndarray): Array of values.
    weights (numpy.ndarray): Array of weights corresponding to the values.

    Returns:
    float: Weighted standard deviation.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return np.sqrt(variance)
