import time

import geopandas as gpd
import osmnx
import pandas as pd
import pytest

import pandana2
from pandana2 import dijkstra


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
    edges = pd.concat([simple_graph, simple_graph_reverse])
    nodes = pd.DataFrame(index=["a", "b", "c", "d", "e", "f"])
    network = pandana2.PandanaNetwork(edges=edges, nodes=nodes)
    network.preprocess(weight_cutoff=1.2)
    return network


def test_basic_edges(simple_graph):
    assert simple_graph.edges.to_dict(orient="records") == [
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
    decay_func = pandana2.linear_decay(0.5)
    values_df = pd.DataFrame({"value": [1, 2, 3]}, index=["b", "d", "c"])
    aggregations_series = simple_graph.aggregate(
        values_df=values_df,
        decay_func=decay_func,
        aggregation="sum",
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
    values_df = pd.DataFrame({"value": [1, 2, 3]}, index=["b", "d", "c"])
    aggregations_series = simple_graph.aggregate(
        values_df=values_df,
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


@pytest.fixture()
def redfin_df():
    df = pd.read_csv("tests/data/redfin_2025-04-04-13-35-42.csv")
    return gpd.GeoDataFrame(
        df[["$/SQUARE FEET"]],
        geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE),
        crs="EPSG:4326",
    )


def test_home_price_aggregation(redfin_df):
    """
    pandana2.PandanaNetwork.from_osmnx_local_streets_from_place_query(
        "Oakland, CA"
    ).write("pandana_oakland.pickle")
    """

    net = pandana2.PandanaNetwork.read("pandana_oakland.pickle")

    redfin_df["node_id"] = net.nearest_nodes(redfin_df)
    redfin_df["ones"] = 1
    assert redfin_df.node_id.isin(net.nodes.index).all()

    t0 = time.time()
    net.preprocess(weight_cutoff=1500)
    print("Finished dijkstra in {:.2f} seconds".format(time.time() - t0))

    t0 = time.time()
    nodes = net.nodes.copy()
    nodes["average price/sqft"] = net.aggregate(
        redfin_df,
        decay_func=pandana2.no_decay(1500),
        value_col="$/SQUARE FEET",
        aggregation="mean",
    )
    nodes["count"] = net.aggregate(
        redfin_df,
        decay_func=pandana2.no_decay(1500),
        value_col="ones",
        aggregation="sum",
    )
    nodes["count"] = nodes["count"].fillna(0)
    print("Finished aggregation in {:.2f} seconds".format(time.time() - t0))

    # plot the output
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(24, 16))
    net.edges.plot(ax=ax, color="grey", linewidth=1, zorder=0)
    nodes.plot(column="count", markersize=1, ax=ax, legend=True, cmap="Reds", zorder=2)
    redfin_df.plot(
        column="$/SQUARE FEET",
        markersize=1,
        ax=ax,
        legend=True,
        cmap="Greens",
        zorder=3,
    )
    plt.savefig("pandana_test_plot.svg", bbox_inches="tight")
