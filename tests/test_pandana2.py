import time

import geopandas as gpd
import numpy as np
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
    network = pandana2.PandanaNetwork(
        edges=edges,
        nodes=nodes,
        from_nodes_col="from",
        to_nodes_col="to",
        edge_costs_col="edge_cost",
    )
    network.preprocess(
        weight_cutoff=1.2,
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
    decay_func = pandana2.LinearDecay(0.5)
    values = pd.Series([1, 2, 3], index=["b", "d", "c"])
    aggregations_series = simple_graph.aggregate(
        values=values,
        decay_func=decay_func,
        aggregation="sum",
    )
    assert aggregations_series.round(2).to_dict() == {
        "a": round(2 * 0.2 / 0.5 + 3 * 0.3 / 0.5, 2),
        "b": 1.0,
        "c": 3 + 2 * 0.4 / 0.5,
        "d": 2 + 3 * 0.4 / 0.5,
    }


def test_flat_aggregation(simple_graph):
    values = pd.Series([1, 2, 3], index=["b", "d", "c"])
    aggregations_series = simple_graph.aggregate(
        values=values,
        decay_func=pandana2.NoDecay(0.5),
        aggregation="sum",
    )
    assert aggregations_series.to_dict() == {
        "a": 5,
        "b": 1,
        "c": 5,
        "d": 5,
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
    pandana2.PandanaNetwork.from_osmnx_local_streets_place_query(
        "Oakland, CA"
    ).write(edges_filename=edges_filename, nodes_filename=nodes_filename)
    """

    with pytest.raises(Exception) as e:
        pandana2.PandanaNetwork.read(
            edges_filename=edges_filename,
            nodes_filename=nodes_filename,
            edge_costs_col="foobar",
        )
    assert "edge_costs_col='foobar' not found in edges DataFrame" in str(e)

    net = pandana2.PandanaNetwork.read(
        edges_filename=edges_filename,
        nodes_filename=nodes_filename,
    )

    redfin_df["node_id"] = net.nearest_nodes(redfin_df)
    assert redfin_df.node_id.isin(net.nodes.index).all()

    t0 = time.time()
    net.preprocess(weight_cutoff=1500)
    print("Finished dijkstra in {:.2f} seconds".format(time.time() - t0))

    with pytest.raises(Exception) as e:
        net.aggregate(
            values=pd.Series(1, index=redfin_df["node_id"]),
            decay_func=pandana2.NoDecay(2000),
            aggregation="sum",
        )
    assert "Decay function has a max weight greater than the value" in str(e)

    with pytest.raises(Exception) as e:
        net.aggregate(
            values=pd.Series(1, index=["does not exist"]),
            decay_func=pandana2.NoDecay(500),
            aggregation="sum",
        )
    assert "Values should have an index which maps to the nodes DataFrame" in str(e)

    t0 = time.time()
    nodes = net.nodes.copy()
    test_osm_id = 53057774

    # count observations within 500 meters
    nodes["count"] = net.aggregate(
        values=pd.Series(1, index=redfin_df["node_id"]),
        decay_func=pandana2.NoDecay(500),
        aggregation="sum",
    )
    print("Finished first aggregation in {:.2f} seconds".format(time.time() - t0))
    assert nodes["count"].loc[test_osm_id] == 2.0

    # count observations within 1000 meters
    nodes["count"] = net.aggregate(
        values=pd.Series(1, index=redfin_df["node_id"]),
        decay_func=pandana2.NoDecay(1000),
        aggregation="sum",
    )
    assert nodes["count"].loc[test_osm_id] == 4.0

    # count observations within 1500 meters
    nodes["count"] = net.aggregate(
        values=pd.Series(1, index=redfin_df["node_id"]),
        decay_func=pandana2.NoDecay(1500),
        aggregation="sum",
    )
    assert nodes["count"].loc[test_osm_id] == 14.0

    # now let's find the 4 observations within 1000 meters
    # these are the nodes within 1000 meters
    filtered_min_weights_df = net.min_weights_df[
        (net.min_weights_df["from"] == test_osm_id)
        & (net.min_weights_df.weight <= 1000)
    ]
    # these are the redfin observations within 1000 meters
    filtered_redfin_df = redfin_df[
        redfin_df.node_id.isin(filtered_min_weights_df["to"])
    ]
    assert pd.Series(
        filtered_redfin_df["$/SQUARE FEET"].values,
        index=filtered_redfin_df["node_id"].values,
    ).to_dict() == {53061872: 543.0, 53148507: 821.0, 53098112: 806.0, 53148506: 585.0}

    nodes["average price/sqft"] = net.aggregate(
        values=pd.Series(redfin_df["$/SQUARE FEET"].values, index=redfin_df["node_id"]),
        decay_func=pandana2.NoDecay(1000),
        aggregation="mean",
    )
    expected = (543 + 821 + 806 + 585) / 4
    assert nodes["average price/sqft"].loc[test_osm_id] == expected

    nodes["average price/sqft"] = net.aggregate(
        values=pd.Series(redfin_df["$/SQUARE FEET"].values, index=redfin_df["node_id"]),
        decay_func=pandana2.LinearDecay(1000),
        aggregation="mean",
    )
    expected = np.average(
        [543, 821, 806, 585],
        weights=[
            (1000 - 947.10) / 1000,
            (1000 - 148.65) / 1000,
            (1000 - 856.11) / 1000,
            (1000 - 226.53) / 1000,
        ],
    )
    assert round(nodes["average price/sqft"].loc[test_osm_id], 4) == round(expected, 4)

    nodes["min price/sqft"] = net.aggregate(
        values=pd.Series(redfin_df["$/SQUARE FEET"].values, index=redfin_df["node_id"]),
        decay_func=pandana2.NoDecay(1000),
        aggregation="min",
    )
    assert nodes["min price/sqft"].loc[test_osm_id] == 543

    # weights should be ignored
    nodes["min price/sqft"] = net.aggregate(
        values=pd.Series(redfin_df["$/SQUARE FEET"].values, index=redfin_df["node_id"]),
        decay_func=pandana2.LinearDecay(1000),
        aggregation="min",
    )
    assert nodes["min price/sqft"].loc[test_osm_id] == 543

    nodes["max price/sqft"] = net.aggregate(
        values=pd.Series(redfin_df["$/SQUARE FEET"].values, index=redfin_df["node_id"]),
        decay_func=pandana2.NoDecay(1000),
        aggregation="max",
    )
    assert nodes["max price/sqft"].loc[test_osm_id] == 821

    # weights should be ignored
    nodes["max price/sqft"] = net.aggregate(
        values=pd.Series(redfin_df["$/SQUARE FEET"].values, index=redfin_df["node_id"]),
        decay_func=pandana2.LinearDecay(1000),
        aggregation="max",
    )
    assert nodes["max price/sqft"].loc[test_osm_id] == 821

    nodes["median price/sqft"] = net.aggregate(
        values=pd.Series(redfin_df["$/SQUARE FEET"].values, index=redfin_df["node_id"]),
        decay_func=pandana2.NoDecay(1000),
        aggregation="median",
    )
    assert nodes["median price/sqft"].loc[test_osm_id] == 585

    nodes["median price/sqft"] = net.aggregate(
        values=pd.Series(redfin_df["$/SQUARE FEET"].values, index=redfin_df["node_id"]),
        decay_func=pandana2.LinearDecay(1000),
        aggregation="median",
    )
    assert nodes["median price/sqft"].loc[test_osm_id] == 806

    nodes["std price/sqft"] = net.aggregate(
        values=pd.Series(redfin_df["$/SQUARE FEET"].values, index=redfin_df["node_id"]),
        decay_func=pandana2.NoDecay(1000),
        aggregation="std",
    )
    assert round(nodes["std price/sqft"].loc[test_osm_id], 2) == 125.74

    nodes["std price/sqft"] = net.aggregate(
        values=pd.Series(redfin_df["$/SQUARE FEET"].values, index=redfin_df["node_id"]),
        decay_func=pandana2.LinearDecay(1000),
        aggregation="std",
    )
    assert round(nodes["std price/sqft"].loc[test_osm_id], 2) == 118.02

    nodes["average price/sqft"] = net.aggregate(
        values=pd.Series(redfin_df["$/SQUARE FEET"].values, index=redfin_df["node_id"]),
        decay_func=pandana2.ExponentialDecay(max_weight=1000, flatness_param=1),
        aggregation="mean",
    )
    expected = np.average(
        [543, 821, 806, 585],
        weights=[
            np.exp(-1 * 947.10 / 1000),
            np.exp(-1 * 148.65 / 1000),
            np.exp(-1 * 856.11 / 1000),
            np.exp(-1 * 226.53 / 1000),
        ],
    )
    assert round(nodes["average price/sqft"].loc[test_osm_id], 4) == round(expected, 4)

    nodes["average price/sqft"] = net.aggregate(
        values=pd.Series(redfin_df["$/SQUARE FEET"].values, index=redfin_df["node_id"]),
        decay_func=pandana2.ExponentialDecay(max_weight=1000, flatness_param=2),
        aggregation="mean",
    )
    expected = np.average(
        [543, 821, 806, 585],
        weights=[
            np.exp(-1 * 947.10 / 1000 * 2),
            np.exp(-1 * 148.65 / 1000 * 2),
            np.exp(-1 * 856.11 / 1000 * 2),
            np.exp(-1 * 226.53 / 1000 * 2),
        ],
    )
    assert round(nodes["average price/sqft"].loc[test_osm_id], 4) == round(expected, 4)

    out_df = net.aggregate(
        values=pd.Series(redfin_df["$/SQUARE FEET"].values, index=redfin_df["node_id"]),
        decay_func=pandana2.LinearDecay(max_weight=1000),
        aggregation={
            "mean_price": "mean",
            "median_price": "median",
            "min_price": "min",
            "max_price": "max",
        },
    )
    assert out_df.loc[test_osm_id].round(2).to_dict() == {
        "max_price": 821,
        "mean_price": 711.53,
        "median_price": 806,
        "min_price": 543,
    }
