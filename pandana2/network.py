import geopandas as gpd
import networkx as nx
import pandas as pd
import shapely


def make_edges(
    graph: nx.Graph, max_weight: float, weight_col: str = "length"
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
        ],
    )


def nearest_nodes(
    values_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Map each point in values_gdf to its nearest node in nodes_gdf
    :param values_gdf: A GeoDataFrame (usually points) with columns for values (e.g. a
        GeoDataFrame of amenity locations, or population or jobs
    :param nodes_gdf: The return value of pandana2.make_nodes
    :return: A GeoDataFrame similar to values_gdf which now contains a node_id column, suitable
        to pass to pandana2.aggregate
    """
    return (
        values_gdf.to_crs(epsg=3857)
        .sjoin_nearest(nodes_gdf.to_crs(epsg=3857))
        .rename(columns={"index_right": "node_id"})
        .set_index("node_id")
    )


def make_nodes(
    graph: nx.Graph, x_col: str = "x", y_col: str = "y", crs: int = 4326
) -> gpd.GeoDataFrame:
    """
    Given a graph, return a GeoDataFrame of point geometry where the index is node_ids.  This
        is used by nearest nodes (to map a variable to the network), but isn't strictly required

    :param graph: a networkx graph, e.g. from osmnx.graph_from_place("Oakland, CA", network_type="walk")
    :param x_col: attribute of each node which contains the x value (osmnx uses 'x')
    :param y_col:  attribute of each node which contains the y value (osmnx uses 'y')
    :param crs: the coordinate reference system of the GeoDataFrame (default is lat-lng)
    :return:
    """
    return gpd.GeoDataFrame(
        geometry=[
            shapely.Point(graph.nodes[node][x_col], graph.nodes[node][y_col])
            for node in graph.nodes
        ],
        index=[node for node in graph.nodes],
        crs=crs,
    )
