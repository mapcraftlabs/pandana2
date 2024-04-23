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
        if from_node in seen or from_node not in indexes:
            continue

        seen.add(from_node)
        ind = indexes[from_node]

        while ind < len(from_nodes) and from_nodes[ind] == from_node:
            to_node, cost = to_nodes[ind], edge_weights[ind]
            ind += 1
            if to_node in seen:
                continue

            prev_weight = min_weights.get(to_node, None)
            new_weight = current_cost + cost
            if new_weight > cutoff:
                continue

            if prev_weight is None or new_weight < prev_weight:
                min_weights[to_node] = new_weight
                heappush(q, (new_weight, to_node))

    return min_weights
