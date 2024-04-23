import pandas as pd
from pandana2 import dijkstra


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
        columns=["from", "to", "weight"],
    )
    edges["weight"] = edges["weight"].astype("float64")

    assert dict(
        dijkstra(edges["from"].values, edges["to"].values, edges.weight.values, 1, 15)
    ) == {1: 0, 2: 7, 4: 5, 5: 14, 6: 11, 3: 15, 7: 22}
    assert dict(
        dijkstra(edges["from"].values, edges["to"].values, edges.weight.values, 6, 15)
    ) == {6: 0, 7: 11}
