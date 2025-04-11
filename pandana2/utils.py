from typing import Literal

import numpy as np
import pandas as pd


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


Aggregation = Literal["max", "mean", "median", "min", "std", "sum"]


def do_single_aggregation(
    merged_df: pd.DataFrame,
    values_col: str,
    origin_node_id_col: str,
    decayed_weights_col: str,
    aggregation: Aggregation,
):
    if aggregation in ["median", "std"]:
        lambda_func = {
            "median": weighted_median,
            "std": weighted_std,
        }[aggregation]
        return merged_df.groupby(origin_node_id_col).apply(
            lambda group: lambda_func(
                group[values_col].values, weights=group[decayed_weights_col].values
            ),
            include_groups=False,
        )

    if aggregation in ["min", "max"]:
        # do not every apply weights for min / max
        decayed_weights = 1
    else:
        decayed_weights = merged_df[decayed_weights_col]

    def do_aggregation(values: pd.Series, _aggregation: Aggregation):
        return values.groupby(merged_df[origin_node_id_col]).agg(_aggregation)

    if aggregation == "mean":
        # could do this with np.average, but it should be faster to do it with 2
        # sums than a .apply like the median below
        sum_of_values = do_aggregation(merged_df[values_col] * decayed_weights, "sum")
        sum_of_weights = do_aggregation(decayed_weights, "sum")
        return sum_of_values / sum_of_weights

    return do_aggregation(merged_df[values_col] * decayed_weights, aggregation)
