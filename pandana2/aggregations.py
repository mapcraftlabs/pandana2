import numba
from numba.typed import Dict, List
from numba.types import float64, int64, ListType, DictType
import numpy as np
import pandas as pd
from typing import Union
from pandana2.dijkstra import _dijkstra_all_pairs


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
    return lambda x: x[x[weight_col] <= max_weight][value_col].agg(aggregation).round(3)


def linear_decay_aggregation(
    aggregation_func=np.sum,
):
    """
    Network aggregations with linear decay.  Values will be filtered out where weight_col
        is greater than max_weight, then value_col is aggregated using the passed aggregation
        method.  Linear decay means a value at max_weight will be weighted as zero while a value
        at the origin node is weighted at 1 and a weight halfway to max_weight will be weighted
        as 0.5.
    :param aggregation_func: The aggregation to use, can be anything which pandas.DataFrame.agg accepts
    :return: A value for the given origin node
    """

    @numba.jit(float64(float64[:], float64[:], float64))
    def func(values, weights, cutoff):
        return aggregation_func(values * np.clip(cutoff - weights, 0, None) / cutoff)

    return func


def aggregate(
    values_df: pd.Series,
    all_nodes_ids: pd.Index,
    edges_df: pd.DataFrame,
    cutoff: float,
    agg_func,
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
    :return: A series indexes by all the origin node ids in edges_df with values returned
        by group_func
    """
    edges_df = edges_df.reset_index()

    values_dict = Dict.empty(key_type=int64, value_type=ListType(float64))
    for node_id, value in values_df.items():
        if node_id not in values_dict:
            values_dict[node_id] = List(lsttype=ListType(float64))
        values_dict[node_id].append(float(value))

    return pd.Series(
        _aggregate(
            all_nodes_ids.values,
            values_dict,
            edges_df["u"].values,
            edges_df["v"].values,
            edges_df["length"].values,
            cutoff,
            agg_func,
        ),
        index=all_nodes_ids,
    )


@numba.jit(
    float64[:](
        int64[:],
        DictType(int64, ListType(float64)),
        int64[:],
        int64[:],
        float64[:],
        float64,
        float64(float64[:], float64[:], float64).as_type(),
    )
)
def _aggregate(
    node_ids: np.array,  # all node_ids
    values_dict: dict,  # keys are node ids and values are a list of values
    from_nodes: np.array,  # start node id of edges (ints)
    to_nodes: np.array,  # end node_id of edges (ints)
    edge_costs: np.array,  # weights (floats)
    cutoff: float64,
    agg_func: float64(float64[:], float64[:], float64),
):
    ret = np.empty(len(node_ids), dtype="float64")
    all_min_weights = _dijkstra_all_pairs(from_nodes, to_nodes, edge_costs, cutoff)
    for i in range(len(node_ids)):
        from_node_id = node_ids[i]
        if from_node_id not in all_min_weights:
            ret[i] = np.nan
            continue

        min_weights = all_min_weights[from_node_id]
        values, weights = [], []
        for node_id, weight in min_weights.items():
            if node_id in values_dict:
                for value in values_dict[node_id]:
                    values.append(value)
                    weights.append(weight)
        values, weights = np.array(values), np.array(weights)

        ret[i] = agg_func(values, weights, cutoff)
    return ret
