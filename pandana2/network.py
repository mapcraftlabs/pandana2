import geopandas as gpd
import pandas as pd
import osmnx
import pickle
from typing import Callable

from pandana2.dijkstra import dijkstra_all_pairs


class PandanaNetwork:
    edges: gpd.GeoDataFrame | pd.DataFrame
    nodes: gpd.GeoDataFrame | pd.DataFrame
    min_weights_df: pd.DataFrame = None
    cutoff: float = None

    def __init__(
        self,
        edges: gpd.GeoDataFrame | pd.DataFrame,
        nodes: gpd.GeoDataFrame | pd.DataFrame,
    ):
        self.edges = edges
        self.nodes = nodes

    def preprocess(
        self,
        weight_cutoff: float,
        from_nodes_col: str = "u",
        to_nodes_col: str = "v",
        edge_costs_col: str = "length",
    ):
        self.min_weights_df = dijkstra_all_pairs(
            self.edges.reset_index(),
            cutoff=weight_cutoff,
            from_nodes_col=from_nodes_col,
            to_nodes_col=to_nodes_col,
            edge_costs_col=edge_costs_col,
        )
        self.cutoff = weight_cutoff

    def nearest_nodes(self, values_gdf: gpd.GeoDataFrame) -> pd.Series:
        """
        Map each point in values_gdf to its nearest node in nodes_gdf
        :param values_gdf: A GeoDataFrame (usually points) with columns for values (e.g. a
            GeoDataFrame of amenity locations, or population or jobs
        :return: A series with the same index as values_gdf with values that come from the
            nodes GeoDataFrame of this network, i.e. the id of closest node for each row in
            values_gdf
        """
        joined_gdf = values_gdf.sjoin_nearest(self.nodes)
        if "index_right" in joined_gdf.columns:
            # older versions of geopandas call it index_right
            return joined_gdf["index_right"]
        return joined_gdf[self.nodes.index.name]

    def aggregate(
        self,
        values_df: pd.DataFrame,
        decay_func: Callable[[pd.Series], pd.Series],
        aggregation: str,
        weight_col: str = "weight",
        value_col: str = "value",
        values_node_id_col: str = "node_id",
        origin_node_id_col: str = "from",
        destination_node_id_col: str = "to",
    ) -> pd.Series | pd.DataFrame:
        """
        Given a values_df which is indexed by node_id and an edges_df with a weight column,
            merge the edges_df to values_df using the destination node id, group by the
            origin node_id, and perform the aggregation specified by group_func
        :param values_node_id_col:
        :param values_df: Typically returned by pandana2.nearest_nodes
        :param decay_func: Typically one of the aggregation functions in this module, e.g.
            linear_decay_aggregation, but can be customized
        :param value_col: The value attribute from values_df to aggregate
        :param destination_node_id_col: A column in edges_df, usually 'to'
        :param origin_node_id_col: A column in edges_df, usually 'from'
        :param weight_col: A column in edges_df which contains the shortest path weight, usually 'weight'
        :param aggregation: Anything you can pass to `.agg`  i.e. 'sum' or 'np.sum', etc.
        :return: A series indexed by all the origin node ids in edges_df with values returned
            by group_func
        """
        merged_df = self.min_weights_df.merge(
            values_df,
            how="inner",
            left_on=destination_node_id_col,
            right_on=values_node_id_col,
        )
        decayed_weights = decay_func(merged_df[weight_col])
        pd.testing.assert_index_equal(merged_df.index, decayed_weights.index)
        return (
            (decayed_weights * merged_df[value_col])
            .groupby(merged_df[origin_node_id_col])
            .agg(aggregation)
            .round(3)
        )

    def write(self, filename: str):
        """
        Write this object to a pickled file
        """
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def read(filename: str):
        """
        Read a PandanaNetwork from a pickled file
        """
        with open(filename, "rb") as file:
            return pickle.load(file)

    @staticmethod
    def from_osmnx_local_streets_from_place_query(place_query: str):
        """
        Use osmnx to grab local street network using settings appropriate for pandana2
        """
        osmnx.settings.bidirectional_network_types = ["all"]
        graph = osmnx.graph_from_place(
            place_query,
            custom_filter='["highway"~"residential|secondary|tertiary|secondary_link|tertiary_link|unclassified"]',
        )
        nodes, edges = osmnx.graph_to_gdfs(graph)

        return PandanaNetwork(
            edges=edges.reset_index(level=2, drop=True)[["length", "geometry"]],
            nodes=nodes[["geometry"]],
        )
