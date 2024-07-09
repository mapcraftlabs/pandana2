import pandas as pd
from pandana2 import dijkstra_all_pairs


def test_dijkstra_basic():
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
        columns=["from", "to", "edge_costs"],
    )

    results = dijkstra_all_pairs(edges, 15)
    assert results[1] == {1: 0, 2: 7, 4: 5, 5: 14, 6: 11, 3: 15}
    assert results[6] == {6: 0, 7: 11}
