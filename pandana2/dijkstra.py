from collections import defaultdict
from heapq import heappop, heappush
import numpy as np
import pandas as pd


def dijkstra(edge_from: np.array, edge_to: np.array, edge_weight: np.array, f, t):
    g = defaultdict(list)
    for i in range(edge_from.size):
        g[edge_from[i]].append((edge_weight[i], edge_to[i]))

    q, seen, min_weight = [(0, f)], set(), {f: 0}
    while q:
        (cost, v1) = heappop(q)
        if v1 not in seen:
            seen.add(v1)
            if v1 == t:
                return cost

            for c, v2 in g.get(v1, ()):
                if v2 in seen:
                    continue
                prev_weight = min_weight.get(v2, None)
                next_weight = cost + c
                if prev_weight is None or next_weight < prev_weight:
                    min_weight[v2] = next_weight
                    heappush(q, (next_weight, v2))

    return float("inf"), None


if __name__ == "__main__":
    edges = pd.DataFrame(
        [
            ("A", "B", 7),
            ("A", "D", 5),
            ("B", "C", 8),
            ("B", "D", 9),
            ("B", "E", 7),
            ("C", "E", 5),
            ("D", "E", 15),
            ("D", "F", 6),
            ("E", "F", 8),
            ("E", "G", 9),
            ("F", "G", 11),
        ],
        columns=["from", "to", "weight"],
    )

    print("=== Dijkstra ===")
    print(edges)
    assert (
        dijkstra(
            edges["from"].values, edges["to"].values, edges.weight.values, "A", "E"
        )
        == 14
    )
    assert (
        dijkstra(
            edges["from"].values, edges["to"].values, edges.weight.values, "F", "G"
        )
        == 11
    )
