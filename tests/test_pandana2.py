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
    edges = pd.concat([simple_graph, simple_graph_reverse])
    nodes = pd.DataFrame(index=["a", "b", "c", "d", "e", "f"])
    network = pandana2.PandanaNetwork(edges=edges, nodes=nodes)
    network.preprocess(
        weight_cutoff=1.2,
        from_nodes_col="from",
        to_nodes_col="to",
        edge_costs_col="edge_cost",
    )
    return network


def test_basic_edges(simple_graph):
    assert simple_graph.min_weights_df.to_dict(orient="records") == [
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
    values = pd.Series([1, 2, 3], index=["b", "d", "c"])
    aggregations_series = simple_graph.aggregate(
        values=values,
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
    values = pd.Series([1, 2, 3], index=["b", "d", "c"])
    aggregations_series = simple_graph.aggregate(
        values=values,
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
    nodes_filename = "tests/data/nodes.parquet"
    edges_filename = "tests/data/edges.parquet"

    """
    # uncomment to refresh the test data
    pandana2.PandanaNetwork.from_osmnx_local_streets_from_place_query(
        "Oakland, CA"
    ).write(edges_filename=edges_filename, nodes_filename=nodes_filename)
    """

    net = pandana2.PandanaNetwork.read(
        edges_filename=edges_filename, nodes_filename=nodes_filename
    )

    redfin_df["node_id"] = net.nearest_nodes(redfin_df)
    assert redfin_df.node_id.isin(net.nodes.index).all()

    t0 = time.time()
    net.preprocess(weight_cutoff=1500)
    print("Finished dijkstra in {:.2f} seconds".format(time.time() - t0))

    t0 = time.time()
    nodes = net.nodes.copy()
    nodes["average price/sqft"] = net.aggregate(
        values=pd.Series(redfin_df["$/SQUARE FEET"], index=redfin_df["node_id"]),
        decay_func=pandana2.no_decay(1500),
        aggregation="mean",
    )
    nodes["count"] = net.aggregate(
        values=pd.Series(1, index=redfin_df["node_id"]),
        decay_func=pandana2.no_decay(1500),
        aggregation="sum",
    )
    print("Finished aggregation in {:.2f} seconds".format(time.time() - t0))
