from typing import Callable, Union

import pandas as pd


def no_decay_aggregation(
    max_weight: float, value_col: str, aggregation: str, weight_col="weight"
):
    """
    Network aggregations with no decay.  Values will be filtered out where weight_col
        is greater than max_weight, then value_col is aggregated using the passed aggregation
        method.  No decay means a value at max_weight will be weighted equally to a value
        at the origin node.
    :param max_weight: Values beyond max_weight (sum of weight_col in network distance)
        will not be considered
    :param value_col: The column in values_df to aggregate
    :param aggregation: The aggregation to use, can be anything which pandas.DataFrame.agg accepts
    :param weight_col: The column in edges_df to use as the weight column
    :return: A value for the given origin node
    """
    return (
        lambda df, group_by_col: (df[value_col] * (df[weight_col] < max_weight))
        .groupby(df[group_by_col])
        .agg(aggregation)
        .round(3)
    )


def linear_decay_aggregation(
    max_weight: float, value_col: str, aggregation: str, weight_col="weight"
):
    """
    Network aggregations with linear decay.  Values will be filtered out where weight_col
        is greater than max_weight, then value_col is aggregated using the passed aggregation
        method.  Linear decay means a value at max_weight will be weighted as zero while a value
        at the origin node is weighted at 1 and a weight halfway to max_weight will be weighted
        as 0.5.
    :param max_weight: Values beyond max_weight (sum of weight_col in network distance)
        will not be considered
    :param value_col: The column in values_df to aggregate
    :param aggregation: The aggregation to use, can be anything which pandas.DataFrame.agg accepts
    :param weight_col: The column in edges_df to use as the weight column
    :return: A value for the given origin node
    """
    return (
        lambda df, group_by_col: (
            df[value_col] * (max_weight - df[weight_col]).clip(lower=0) / max_weight
        )
        .groupby(df[group_by_col])
        .agg(aggregation)
        .round(3)
    )


def aggregate(
    values_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    group_func: Callable[[pd.DataFrame, str], pd.DataFrame],
    origin_node_id_col: str = "from",
    destination_node_id_col: str = "to",
) -> Union[pd.Series, pd.DataFrame]:
    """
    Given a values_df which is indexed by node_id and an edges_df with a weight column,
        merge the edges_df to values_df using the destination node id, group by the
        origin node_id, and perform the aggregation specified by group_func
    :param values_df: Typically returned by pandana2.nearest_nodes
    :param edges_df: Typically returned by pandana2.make_edges
    :param group_func: Typically one of the aggregation functions in this module, e.g.
        linear_decay_aggregation, but can be customized
    :param destination_node_id_col: a column in edges_df, usually 'to'
    :param origin_node_id_col: a column in edges_df, usually 'from'
    :return: A series indexed by all the origin node ids in edges_df with values returned
        by group_func
    """
    merged_df = edges_df.merge(
        values_df, how="inner", left_on=destination_node_id_col, right_index=True
    )
    return group_func(merged_df, origin_node_id_col)
