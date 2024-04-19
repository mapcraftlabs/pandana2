import geopandas as gpd
import networkx as nx
import pandas as pd
import shapely


def nodes_and_edges(graph: nx.DiGraph, max_weight: float, weight_col: str = "length") -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Given a graph, return a DataFrame of weights for all combinations of nodes within max_weight of
        each other.

    :param graph: a networkx graph, e.g. from osmnx.graph_from_place("Oakland, CA", network_type="walk")
    :param max_weight: keep nodes within this weight of each origin node
    :param weight_col: the attribute on the networkx edges to use as the weight (osmnx uses "length")
    :return: a tuple whose first element is a GeoDataFrame of node ids and positions and the second element
        if a DataFrame of "from", "to", and "weight" which contains all node combinations that are within
        max_weight of each other
    """
    nodes_gdf = gpd.GeoDataFrame(
        geometry=[
            shapely.Point(graph.nodes[node]["x"], graph.nodes[node]["y"])
            for node in graph.nodes
        ],
        index=[node for node in graph.nodes],
        crs=4326,
    )

    weights = nx.all_pairs_dijkstra_path_length(
        graph, cutoff=max_weight, weight=weight_col
    )
    edges_df = pd.DataFrame.from_records(
        [
            {
                "from": from_node_id,
                "to": to_node_id,
                "weight": weight,
            }
            for from_node_id, to_dict in weights
            for to_node_id, weight in to_dict.items()
            if from_node_id != to_node_id
        ],
    )

    return nodes_gdf, edges_df
