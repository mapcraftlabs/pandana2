from heapq import heappop, heappush
import numba
import pandas as pd
from numba.types import int64, float64, DictType
import numpy as np

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
    (DictType(int64, DictType(int64, float64)))(int64[:], int64[:], float64[:], float64)
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

    indexes = DictType.empty(key_type=int64, value_type=int64)
    for i in range(len(from_nodes)):
        assert edge_costs[i] > 0, "Edge costs cannot be negative"
        if i > 1:
            # we require from_nodes to be sorted
            assert from_nodes[i] >= from_nodes[i - 1], "from_nodes must be sorted"
        if from_nodes[i] not in indexes:
            # indexes[node_id] holds the first array index that from_node is seen in from_nodes
            indexes[from_nodes[i]] = i

    results = DictType.empty(
        key_type=int64, value_type=DictType.empty(key_type=int64, value_type=float64)
    )

    for from_node in indexes.keys():
        results[from_node] = _dijkstra(
            from_nodes, to_nodes, edge_costs, from_node, cutoff, indexes
        )

    return results


def dijkstra_all_pairs(
    df: pd.DataFrame,
    cutoff: float,  # cutoff weight (float)
    from_nodes_col="from",
    to_nodes_col="to",
    edge_costs_col="edge_cost",
) -> dict[int, dict[int, float]]:
    """
    Same as above, but pass in a DataFrame
    Should have from_nodes_col, to_nodes_col, and edge_costs_col as columns
    """
    return _dijkstra_all_pairs(
        df[from_nodes_col].values,
        df[to_nodes_col].values,
        df[edge_costs_col].astype("float").values,
        cutoff,
    )


def dijkstra_all_pairs_df(
    df: pd.DataFrame,
    cutoff: float,  # cutoff weight (float)
    from_nodes_col="from",
    to_nodes_col="to",
    edge_costs_col="edge_cost",
) -> pd.DataFrame:
    all_unique = set(df[from_nodes_col].unique()) | set(df[to_nodes_col].unique())
    node_id_to_index = {k: v for v, k in enumerate(all_unique)}
    index_to_node_id = {v: k for k, v in node_id_to_index.items()}
    df[from_nodes_col] = df[from_nodes_col].map(node_id_to_index)
    df[to_nodes_col] = df[to_nodes_col].map(node_id_to_index)
    results = dijkstra_all_pairs(
        df.sort_values(by=[from_nodes_col, to_nodes_col]),
        cutoff,
        from_nodes_col=from_nodes_col,
        to_nodes_col=to_nodes_col,
        edge_costs_col=edge_costs_col,
    )
    ret_df = pd.DataFrame.from_records(
        [
            {"from": from_node, "to": to_node, "min_cost": min_cost}
            for from_node, min_costs in results.items()
            for to_node, min_cost in min_costs.items()
        ]
    )
    ret_df["from"] = ret_df["from"].map(index_to_node_id)
    ret_df["to"] = ret_df["to"].map(index_to_node_id)
    ret_df.sort_values(by=["from", "min_cost"], inplace=True)
    ret_df.rename(columns={"min_cost": "weight"}, inplace=True)
    ret_df["weight"] = ret_df.weight.round(2)
    return ret_df
