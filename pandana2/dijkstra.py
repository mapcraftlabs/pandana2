from heapq import heappop, heappush

import numba
import numpy as np
import pandas as pd
from numba.types import DictType, Tuple, float64, int64

# early code (heavily modified) from https://gist.github.com/kachayev/5990802


@numba.jit(
    (DictType(int64, float64))(
        int64[:],
        int64[:],
        float64[:],
        int64,
        float64,
        DictType(int64, int64),
    )
)
def _dijkstra(
    from_nodes: np.array,  # node ids (ints)
    to_nodes: np.array,  # node ids (ints)
    edge_costs: np.array,  # weights (floats)
    source: int,  # source node
    cutoff: float,  # cutoff weight (float)
    indexes: DictType(int64, int64),  # first occurrence of each node_id in from_nodes
):
    """
    Internal function should not be called except by dijkstra_all_pairs
    """
    # q is the heapq instance
    # seen is a set of which nodes we've seen so far
    # min_weight is a dict where keys are node ids and values are the minimum costs we've seen so far
    q, seen, min_costs = [(0.0, source)], set(), {source: 0.0}
    while q:
        current_cost, from_node = heappop(q)
        if from_node in seen or from_node not in indexes:
            continue

        seen.add(from_node)
        ind = indexes[from_node]

        while ind < len(from_nodes) and from_nodes[ind] == from_node:
            to_node, cost = to_nodes[ind], edge_costs[ind]
            ind += 1
            if to_node in seen:
                continue

            prev_cost = min_costs.get(to_node)
            new_cost = current_cost + cost
            if new_cost > cutoff:
                continue

            if prev_cost is None or new_cost < prev_cost:
                min_costs[to_node] = new_cost
                heappush(q, (new_cost, to_node))

    return min_costs


@numba.jit(
    Tuple((int64[:], int64[:], float64[:]))(int64[:], int64[:], float64[:], float64)
)
def _dijkstra_all_pairs(
    from_nodes: np.array,  # node ids (ints)
    to_nodes: np.array,  # node ids (ints)
    edge_costs: np.array,  # weights (floats)
    cutoff: float,  # cutoff weight (float)
):
    """
    Run dijkstra for every node in from_nodes
    """
    assert (
        len(from_nodes) == len(to_nodes) == len(edge_costs)
    ), "from_nodes, to_nodes, and edge_weights must be same length"

    # indexes[node_id] holds the first array index that from_node is seen in from_nodes
    indexes = DictType.empty(key_type=int64, value_type=int64)
    for i in range(len(from_nodes)):
        assert edge_costs[i] > 0, "Edge costs cannot be negative"
        if i > 1:
            # we require from_nodes to be sorted
            assert from_nodes[i] >= from_nodes[i - 1], "from_nodes must be sorted"
        if from_nodes[i] not in indexes:
            indexes[from_nodes[i]] = i

    # results is a dictionary where keys are "from" nodes and values are dictionaries
    #   the sub-dictionary contains keys which are "to" nodes and the min const between
    #   from and to
    results = DictType.empty(
        key_type=int64, value_type=DictType.empty(key_type=int64, value_type=float64)
    )

    for from_node in indexes.keys():
        results[from_node] = _dijkstra(
            from_nodes, to_nodes, edge_costs, from_node, cutoff, indexes
        )

    # from here down we convert from dictionaries to arrays
    # dictionaries are much more expensive to pass back to python
    total_len = 0
    for to_node_dict in results.values():
        total_len += len(to_node_dict)

    from_nodes = np.empty(total_len, dtype=np.int64)
    to_nodes = np.empty(total_len, dtype=np.int64)
    weights = np.empty(total_len, dtype=np.float64)

    i = 0
    for from_node, to_node_dict in results.items():
        for to_node, weight in to_node_dict.items():
            from_nodes[i] = from_node
            to_nodes[i] = to_node
            weights[i] = weight
            i += 1

    return from_nodes, to_nodes, weights


def dijkstra_all_pairs(
    edges_df: pd.DataFrame,
    cutoff: float,  # cutoff weight (float)
    from_nodes_col="from",
    to_nodes_col="to",
    edge_costs_col="edge_cost",
) -> pd.DataFrame:
    """
    Run dijkstra for every node in the edges DataFrame.  Edges should have from, to, and edge_cost
      columns which can be specified using the optional parameters.  The return value will be node
      from-to connections and the associated weight of the shortest path between them.  Cutoff
      must be passed to keep the result performant and is the maximum weight to consider between
      nearby nodes.
    """
    # it's not the prettiest code, but node ids need to be ints by the time they get into numba
    #  so we translate them here, which is very similar to pd.factorize
    index_to_node_id = pd.Series(
        np.unique([edges_df[from_nodes_col], edges_df[to_nodes_col]])
    )
    node_id_to_index = pd.Series(index_to_node_id.index, index=index_to_node_id.values)
    edges_df[from_nodes_col] = edges_df[from_nodes_col].map(node_id_to_index)
    edges_df[to_nodes_col] = edges_df[to_nodes_col].map(node_id_to_index)
    edges_df = edges_df.sort_values(by=[from_nodes_col, to_nodes_col])

    from_nodes, to_nodes, weight = _dijkstra_all_pairs(
        edges_df[from_nodes_col].values,
        edges_df[to_nodes_col].values,
        edges_df[edge_costs_col].astype("float").values,
        cutoff,
    )

    ret_df = pd.DataFrame({"from": from_nodes, "to": to_nodes, "weight": weight})
    ret_df["from"] = ret_df["from"].map(index_to_node_id)
    ret_df["to"] = ret_df["to"].map(index_to_node_id)
    ret_df.sort_values(by=["from", "weight"], inplace=True)
    ret_df["weight"] = ret_df["weight"].round(2)

    return ret_df
