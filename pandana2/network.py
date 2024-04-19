import geopandas as gpd
import networkx as nx
import pandas as pd
import shapely


def make_edges(
    graph: nx.DiGraph, max_weight: float, weight_col: str = "length"
) -> pd.DataFrame:
    """
    Return a DataFrame of weights (or distances) for all combinations of nodes within max_weight of
        each other.

    :param graph: a networkx graph, e.g. from osmnx.graph_from_place("Oakland, CA", network_type="walk")
    :param max_weight: keep nodes within this weight of each origin node
    :param weight_col: the attribute on the networkx edges to use as the weight (osmnx uses "length")
    :return: a DataFrame of "from", "to", and "weight" which contains all node combinations that are within
        max_weight of each other
    """
    return pd.DataFrame.from_records(
        [
            {
                "from": from_node_id,
                "to": to_node_id,
                "weight": round(weight, 3),
            }
            for from_node_id, to_dict in nx.all_pairs_dijkstra_path_length(
                graph, cutoff=max_weight, weight=weight_col
            )
            for to_node_id, weight in to_dict.items()
            if from_node_id != to_node_id
        ],
    )


def make_nodes(graph: nx.DiGraph, x_col="x", y_col="y", crs=4326) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        geometry=[
            shapely.Point(graph.nodes[node][x_col], graph.nodes[node][y_col])
            for node in graph.nodes
        ],
        index=[node for node in graph.nodes],
        crs=crs,
    )
