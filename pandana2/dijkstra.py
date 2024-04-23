from heapq import heappop, heappush
import numba
from numba.types import int64, float64, DictType
import numpy as np
import pandas as pd


@numba.jit((DictType(int64, float64))(int64[:], int64[:], float64[:], int64, float64))
def dijkstra(
    from_nodes: np.array,  # node ids (ints)
    to_nodes: np.array,  # node ids (ints)
    edge_weights: np.array,  # weights (floats)
    source: int,  # source node (str for now, will be int)
    cutoff: float,  # cutoff weight (float)
):
    indexes = {}
    for i in range(len(from_nodes)):
        if i > 1:
            # we require from_nodes to be sorted
            assert from_nodes[i] >= from_nodes[i - 1]
        if from_nodes[i] not in indexes:
            # we assume from_nodes is sorted
            # indexes[node_id] holds the first array index that from_node is seen in from_nodes
            indexes[from_nodes[i]] = i

    # q is the heapq instance
    # seen is a set of which nodes we've seen so far
    # min_weight is a dict where keys are node ids and values are the minimum weights we've seen so far
    q, seen, min_weights = [(0.0, source)], set(), {source: 0.0}
    while q:
        (current_cost, from_node) = heappop(q)
        if from_node in seen:
            continue

        seen.add(from_node)
        if current_cost >= cutoff:
            return min_weights

        if from_node not in indexes:
            continue

        ind = indexes[from_node]
        while ind < len(from_nodes) and from_nodes[ind] == from_node:
            to_node, cost = to_nodes[ind], edge_weights[ind]
            ind += 1
            if to_node in seen:
                continue
            prev_weight = min_weights.get(to_node, None)
            new_weight = current_cost + cost
            if prev_weight is None or new_weight < prev_weight:
                min_weights[to_node] = new_weight
                heappush(q, (new_weight, to_node))

    return min_weights


if __name__ == "__main__":
    edges = pd.DataFrame(
        [
            (1, 2, 7),
            (1, 4, 5),
            (2, 3, 8),
            (2, 4, 9),
            (2, 5, 7),
            (3, 5, 5),
            (4, 5, 15),
            (4, 6, 6),
            (5, 6, 8),
            (5, 7, 9),
            (6, 7, 11),
        ],
        columns=["from", "to", "weight"],
    )
    edges["weight"] = edges["weight"].astype("float64")

    print(edges)
    assert dijkstra(
        edges["from"].values, edges["to"].values, edges.weight.values, 1, 15
    ) == {1: 0, 2: 7, 4: 5, 5: 14, 6: 11, 3: 15, 7: 22}
    assert dijkstra(
        edges["from"].values, edges["to"].values, edges.weight.values, 6, 15
    ) == {6: 0, 7: 11}
