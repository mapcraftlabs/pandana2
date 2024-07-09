import networkx
import osmnx
import pytest

import pandas as pd

import pandana2


@pytest.fixture
def simple_graph():
    """
    From https://networkx.org/documentation/stable/auto_examples/drawing/plot_weighted_graph.html
    """
    simple_graph = pd.DataFrame.from_records(
        [
            {"from": "a", "to": "b", "edge_cost": 0.6},
            {"from": "a", "to": "c", "edge_cost": 0.2},
            {"from": "c", "to": "d", "edge_cost": 0.1},
            {"from": "c", "to": "e", "edge_cost": 0.7},
            {"from": "c", "to": "f", "edge_cost": 0.9},
            {"from": "a", "to": "d", "edge_cost": 0.3},
        ]
    )
    simple_graph_reverse = simple_graph.rename(columns={"from": "to", "to": "from"})
    return pd.concat([simple_graph, simple_graph_reverse])


def test_basic_edges(simple_graph):
    edges = pandana2.dijkstra_all_pairs_df(simple_graph, cutoff=1.2)
    assert edges.to_dict(orient="records") == [
        {"from": "a", "to": "a", "weight": 0.0},
        {"from": "a", "to": "c", "weight": 0.2},
        {"from": "a", "to": "d", "weight": 0.3},
        {"from": "a", "to": "b", "weight": 0.6},
        {"from": "a", "to": "e", "weight": 0.9},
        {"from": "a", "to": "f", "weight": 1.1},
        {"from": "b", "to": "b", "weight": 0.0},
        {"from": "b", "to": "a", "weight": 0.6},
        {"from": "b", "to": "c", "weight": 0.8},
        {"from": "b", "to": "d", "weight": 0.9},
        {"from": "c", "to": "c", "weight": 0.0},
        {"from": "c", "to": "d", "weight": 0.1},
        {"from": "c", "to": "a", "weight": 0.2},
        {"from": "c", "to": "e", "weight": 0.7},
        {"from": "c", "to": "b", "weight": 0.8},
        {"from": "c", "to": "f", "weight": 0.9},
        {"from": "d", "to": "d", "weight": 0.0},
        {"from": "d", "to": "c", "weight": 0.1},
        {"from": "d", "to": "a", "weight": 0.3},
        {"from": "d", "to": "e", "weight": 0.8},
        {"from": "d", "to": "b", "weight": 0.9},
        {"from": "d", "to": "f", "weight": 1.0},
        {"from": "e", "to": "e", "weight": 0.0},
        {"from": "e", "to": "c", "weight": 0.7},
        {"from": "e", "to": "d", "weight": 0.8},
        {"from": "e", "to": "a", "weight": 0.9},
        {"from": "f", "to": "f", "weight": 0.0},
        {"from": "f", "to": "c", "weight": 0.9},
        {"from": "f", "to": "d", "weight": 1.0},
        {"from": "f", "to": "a", "weight": 1.1},
    ]


def test_linear_aggregation(simple_graph):
    edges = pandana2.dijkstra_all_pairs_df(simple_graph, cutoff=1.2)
    group_func = pandana2.linear_decay_aggregation(0.5, "value", "sum")
    values_df = pd.DataFrame({"value": [1, 2, 3]}, index=["b", "d", "c"])
    aggregations_series = pandana2.aggregate(values_df, edges, group_func)
    assert aggregations_series.to_dict() == {
        "a": round(2 * 0.2 / 0.5 + 3 * 0.3 / 0.5, 2),
        "b": 1,
        "c": 3 + 2 * 0.4 / 0.5,
        "d": 2 + 3 * 0.4 / 0.5,
        "e": 0,
        "f": 0,
    }


def test_flat_aggregation(simple_graph):
    edges = pandana2.dijkstra_all_pairs_df(simple_graph, cutoff=1.2)
    group_func = pandana2.no_decay_aggregation(0.5, "value", "sum")
    values_df = pd.DataFrame({"value": [1, 2, 3]}, index=["b", "d", "c"])
    aggregations_series = pandana2.aggregate(values_df, edges, group_func)
    assert aggregations_series.to_dict() == {
        "a": 5,
        "b": 1,
        "c": 5,
        "d": 5,
        "e": 0,
        "f": 0,
    }


def get_amenity_as_dataframe(place_query: str, amenity: str):
    restaurants = osmnx.features_from_place(place_query, {"amenity": amenity})
    restaurants = restaurants.reset_index()
    restaurants = restaurants[restaurants.element_type == "node"]
    restaurants = restaurants[["name", "geometry"]]
    restaurants["count"] = 1
    return restaurants


def test_workflow():
    place_query = "Orinda, CA"
    graph = osmnx.graph_from_place(place_query)
    nodes, edges = osmnx.graph_to_gdfs(graph)

    restaurants_df = get_amenity_as_dataframe(place_query, "restaurant")
    restaurants_df = pandana2.nearest_nodes(restaurants_df, nodes)
    assert restaurants_df.index.isin(nodes.index).all()

    agg_func = pandana2.linear_decay_aggregation()
    import time

    t1 = time.time()
    aggregations_series = pandana2.aggregate(
        restaurants_df["count"], nodes.index, edges, 500, agg_func
    )
    time_diff = time.time() - t1
    print("here", time_diff)
    print(aggregations_series)
    assert aggregations_series.index.isin(nodes.index).all()
    assert aggregations_series.min() >= 0
    assert aggregations_series.max() <= 8
