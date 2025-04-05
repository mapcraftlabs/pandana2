import time

import geopandas as gpd
import osmnx
import pandas as pd
import pytest

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
    edges = pandana2.dijkstra_all_pairs(simple_graph, cutoff=1.2)
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
    edges = pandana2.dijkstra_all_pairs(simple_graph, cutoff=1.2)
    decay_func = pandana2.linear_decay(0.5)
    values_df = pd.DataFrame({"value": [1, 2, 3]}, index=["b", "d", "c"])
    aggregations_series = pandana2.aggregate(
        values_df=values_df, edges_df=edges, decay_func=decay_func, aggregation="sum"
    )
    assert aggregations_series.to_dict() == {
        "a": round(2 * 0.2 / 0.5 + 3 * 0.3 / 0.5, 2),
        "b": 1,
        "c": 3 + 2 * 0.4 / 0.5,
        "d": 2 + 3 * 0.4 / 0.5,
        "e": 0,
        "f": 0,
    }


def test_flat_aggregation(simple_graph):
    edges = pandana2.dijkstra_all_pairs(simple_graph, cutoff=1.2)
    values_df = pd.DataFrame({"value": [1, 2, 3]}, index=["b", "d", "c"])
    aggregations_series = pandana2.aggregate(
        values_df=values_df,
        edges_df=edges,
        decay_func=pandana2.no_decay(0.5),
        aggregation="sum",
    )
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


@pytest.mark.skip()
def test_workflow():
    """
    place_query = "Oakland, CA"
    graph = osmnx.graph_from_place(place_query, network_type="drive")
    nodes, edges = osmnx.graph_to_gdfs(graph)

    print(nodes)
    print(edges)

    nodes[["x", "y"]].to_parquet("nodes.parquet")
    edges.reset_index(level=2, drop=True)[["length"]].to_parquet("edges.parquet")
    """
    edges = pd.read_parquet("edges.parquet")
    nodes = pd.read_parquet("nodes.parquet")
    nodes = gpd.GeoDataFrame(
        nodes,
        geometry=gpd.points_from_xy(nodes.x, nodes.y),
        crs="EPSG:4326",
    ).drop(columns=["x", "y"])

    redfin_df = pd.read_csv("redfin_2025-04-04-13-35-42.csv")
    redfin_df = gpd.GeoDataFrame(
        redfin_df[["$/SQUARE FEET"]],
        geometry=gpd.points_from_xy(redfin_df.LONGITUDE, redfin_df.LATITUDE),
        crs="EPSG:4326",
    )

    redfin_df = pandana2.nearest_nodes(redfin_df, nodes)
    print(redfin_df)
    assert redfin_df.index.isin(nodes.index).all()

    t0 = time.time()
    distances_df = pandana2.dijkstra_all_pairs(
        edges.reset_index(),
        cutoff=1500,
        from_nodes_col="u",
        to_nodes_col="v",
        edge_costs_col="length",
    )
    print("Finished dijkstra_all_pairs in {:.2f} seconds".format(time.time() - t0))
    print(distances_df)

    group_func = pandana2.no_decay(1500, "$/SQUARE FEET", "mean")
    t0 = time.time()
    nodes["average price/sqft"] = pandana2.aggregate(
        redfin_df, distances_df, group_func
    )
    print("Finished aggregation in {:.2f} seconds".format(time.time() - t0))
    print(nodes)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    plt.title("Average Sales Price/SQFT")
    nodes.plot(
        column="average price/sqft",
        markersize=2,
        ax=ax,
        legend=True,
    )
    plt.savefig("average price per sqft.png", dpi=150, bbox_inches="tight")
