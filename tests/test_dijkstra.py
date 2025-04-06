import pandas as pd

from pandana2.dijkstra import dijkstra_all_pairs


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
        columns=["from", "to", "edge_cost"],
    )

    results = dijkstra_all_pairs(edges, 15)
    assert results.to_dict(orient="records") == [
        {"from": 1, "to": 1, "weight": 0.0},
        {"from": 1, "to": 4, "weight": 5.0},
        {"from": 1, "to": 2, "weight": 7.0},
        {"from": 1, "to": 6, "weight": 11.0},
        {"from": 1, "to": 5, "weight": 14.0},
        {"from": 1, "to": 3, "weight": 15.0},
        {"from": 2, "to": 2, "weight": 0.0},
        {"from": 2, "to": 5, "weight": 7.0},
        {"from": 2, "to": 3, "weight": 8.0},
        {"from": 2, "to": 4, "weight": 9.0},
        {"from": 2, "to": 6, "weight": 15.0},
        {"from": 3, "to": 3, "weight": 0.0},
        {"from": 3, "to": 5, "weight": 5.0},
        {"from": 3, "to": 6, "weight": 13.0},
        {"from": 3, "to": 7, "weight": 14.0},
        {"from": 4, "to": 4, "weight": 0.0},
        {"from": 4, "to": 6, "weight": 6.0},
        {"from": 4, "to": 5, "weight": 15.0},
        {"from": 5, "to": 5, "weight": 0.0},
        {"from": 5, "to": 6, "weight": 8.0},
        {"from": 5, "to": 7, "weight": 9.0},
        {"from": 6, "to": 6, "weight": 0.0},
        {"from": 6, "to": 7, "weight": 11.0},
    ]
