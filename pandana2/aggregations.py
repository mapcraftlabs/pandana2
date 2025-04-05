from typing import Callable, Union

import pandas as pd


def no_decay(max_weight: float):
    """
    Network aggregations with no decay.  Values will be filtered out where weight_col
        is greater than max_weight.  No decay means a value at max_weight will be
        weighted equally to a value at the origin node.
    :param max_weight: Values beyond max_weight (sum of weight_col in network distance)
        will not be considered
    :return: A value for the given origin node
    """
    return lambda weights: weights < max_weight


def linear_decay(max_weight: float):
    """
    Network aggregations with linear decay.  Values will be filtered out where weight_col
        is greater than max_weight, then value_col is aggregated using the passed aggregation
        method.  Linear decay means a value at max_weight will be weighted as zero while a value
        at the origin node is weighted at 1 and a weight halfway to max_weight will be weighted
        as 0.5.
    :param max_weight: Values beyond max_weight (sum of weight_col in network distance)
        will not be considered
    :return: A value for the given origin node
    """
    return lambda weights: ((max_weight - weights).clip(lower=0) / max_weight)


def aggregate(
    values_df: pd.DataFrame,
    min_weights_df: pd.DataFrame,
    decay_func: Callable[[pd.Series], pd.Series],
    aggregation: str,
    weight_col: str = "weight",
    value_col: str = "value",
    origin_node_id_col: str = "from",
    destination_node_id_col: str = "to",
) -> Union[pd.Series, pd.DataFrame]:
    """
    Given a values_df which is indexed by node_id and an edges_df with a weight column,
        merge the edges_df to values_df using the destination node id, group by the
        origin node_id, and perform the aggregation specified by group_func
    :param values_df: Typically returned by pandana2.nearest_nodes
    :param min_weights_df: Typically returned by pandana2.dijkstra_all_pairs
    :param decay_func: Typically one of the aggregation functions in this module, e.g.
        linear_decay_aggregation, but can be customized
    :param value_col: The value attribute from values_df to aggregate
    :param destination_node_id_col: A column in edges_df, usually 'to'
    :param origin_node_id_col: A column in edges_df, usually 'from'
    :param weight_col: A column in edges_df which contains the shortest path weight, usually 'weight'
    :param aggregation: Anything you can pass to `.agg`, i.e. 'sum' or np.sum, etc
    :return: A series indexed by all the origin node ids in edges_df with values returned
        by group_func
    """
    merged_df = min_weights_df.merge(
        values_df, how="inner", left_on=destination_node_id_col, right_index=True
    )
    decayed_weights = decay_func(merged_df[weight_col])
    pd.testing.assert_index_equal(merged_df.index, decayed_weights.index)
    return (
        (decayed_weights * merged_df[value_col])
        .groupby(merged_df[origin_node_id_col])
        .agg(aggregation)
        .round(3)
    )
