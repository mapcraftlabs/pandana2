import geopandas as gpd
import osmnx
import pandas as pd

from pandana2.decay_functions import PandanaDecayFunction
from pandana2.dijkstra import dijkstra_all_pairs
from pandana2.utils import Aggregation, do_single_aggregation


class PandanaNetwork:
    edges: gpd.GeoDataFrame | pd.DataFrame
    nodes: gpd.GeoDataFrame | pd.DataFrame
    min_weights_df: pd.DataFrame = None
    weight_cutoff: float = None
    from_nodes_col: str
    to_nodes_col: str
    edge_costs_col: str

    def __init__(
        self,
        edges: gpd.GeoDataFrame | pd.DataFrame,
        nodes: gpd.GeoDataFrame,
        from_nodes_col: str = "u",
        to_nodes_col: str = "v",
        edge_costs_col: str = "length",
    ):
        """
        Pass a DataFrame of edges which is indexed with 'from' and 'to' nodes and has a column to
            represent the edge weight.  The nodes DataFrame contains each of the unique 'from'
            and 'to' node ids with an associated Geometry.  If you use osmnx, the 'from' and 'to'
            columns will be called 'u' and 'v' respectively, and the default edge_costs_col will
            be 'length'.

        :param edges:
        :param nodes:
        :param from_nodes_col: The name of the "from" nodes column (e.g. osmnx uses "u")
        :param to_nodes_col: The name of the "from" nodes column (e.g. osmnx uses "v")
        :param edge_costs_col: The name of the "from" nodes column (e.g. osmnx uses "distance")
            for Euclidian distance, but any impedance (like travel time) could also be used here.
        """
        edges_columns = edges.reset_index().columns
        if from_nodes_col not in edges_columns:
            raise Exception(
                f"from_nodes_col='{from_nodes_col}' not found in edges DataFrame"
            )
        if to_nodes_col not in edges_columns:
            raise Exception(
                f"to_nodes_col='{to_nodes_col}' not found in edges DataFrame"
            )
        if edge_costs_col not in edges_columns:
            raise Exception(
                f"edge_costs_col='{edge_costs_col}' not found in edges DataFrame"
            )
        assert (
            edges.reset_index()[from_nodes_col].isin(nodes.index).all()
        ), "All 'from' node ids should be in the node DataFrame"
        assert (
            edges.reset_index()[to_nodes_col].isin(nodes.index).all()
        ), "All 'to' node ids should be in the node DataFrame"

        self.edges = edges
        self.nodes = nodes
        self.from_nodes_col = from_nodes_col
        self.to_nodes_col = to_nodes_col
        self.edge_costs_col = edge_costs_col

    def preprocess(
        self,
        weight_cutoff: float,
    ):
        """
        Convert the edges DataFrame (which represents the connections in a network), to a "minimum
            weights" DataFrame which contains all the from-to pairs with the shortest path weight
            between from and to nodes, for all pairs such that the minimum weight is less than the
            weight cutoff passed here.  This can be a very expensive operation, but for well-chosen
            networks and cutoffs it should be very fast.
        :param weight_cutoff: Don't investigate from-to pairs whose minimum path is larger than
            this cutoff.
        :return:
        """
        self.min_weights_df = dijkstra_all_pairs(
            self.edges.reset_index(),
            cutoff=weight_cutoff,
            from_nodes_col=self.from_nodes_col,
            to_nodes_col=self.to_nodes_col,
            edge_costs_col=self.edge_costs_col,
        )
        self.weight_cutoff = weight_cutoff

    def nearest_nodes(self, values_gdf: gpd.GeoDataFrame) -> pd.Series:
        """
        Map each point in values_gdf to its nearest node in nodes_gdf

        :param values_gdf: A GeoDataFrame (usually points) with columns for values (e.g. a
            GeoDataFrame of amenity locations, or population or jobs
        :return: A series with the same index as values_gdf with values that come from the
            nodes GeoDataFrame of this network, i.e. the id of closest node for each row in
            values_gdf
        """
        joined_gdf = values_gdf.to_crs(epsg=3857).sjoin_nearest(
            self.nodes.to_crs(epsg=3857)
        )

        if "index_right" in joined_gdf.columns:
            # older versions of geopandas call it index_right
            return joined_gdf["index_right"]

        return joined_gdf[self.nodes.index.name]

    def aggregate(
        self,
        values: pd.Series,
        decay_func: PandanaDecayFunction,
        aggregation: Aggregation | dict[str, Aggregation],
    ) -> pd.Series | pd.DataFrame:
        """
        Perform a network-based aggregation - this is the whole point of this python library.
        :param values: A series where the index is node_ids from the node dataframe and the
            values are floating point values you want to aggregate.  In other words, it's the
            values and the node_ids they are located at.  node_ids can and likely will be
            repeated in the index (i.e. not unique).
        :param decay_func: Typically one of the decay functions in this module, e.g.
            linear_decay, no_decay, etc., and can be customized.
        :param aggregation: Anything you can pass to `.agg`  i.e. 'sum' or 'np.sum', etc.
        :return: A series indexed by all the origin node ids in 'self.nodes' with values computed
            for this aggregation.
        """
        assert isinstance(
            values, pd.Series
        ), "Values should be a Series (see docstring)"

        assert values.index.isin(
            self.nodes.index
        ).all(), "Values should have an index which maps to the nodes DataFrame"

        # these column names are returned by the dijkstra function
        weight_col = "weight"
        origin_node_id_col = "from"
        destination_node_id_col = "to"

        # these column names are just internal to this function
        values_col = "values"
        decayed_weights_col = "decayed_weights"

        if decay_func.max_weight > self.weight_cutoff:
            raise Exception(
                "Decay function has a max weight greater than the value passed to preprocess"
            )

        # for performance, we apply the max_weight filter first
        filtered_weights = self.min_weights_df[
            decay_func.mask(self.min_weights_df[weight_col])
        ]

        merged_df = filtered_weights.merge(
            pd.DataFrame({values_col: values}),
            how="inner",
            left_on=destination_node_id_col,
            right_index=True,
        )

        merged_df[decayed_weights_col] = decay_func.weights(merged_df[weight_col])

        if isinstance(aggregation, dict):
            # support multiple aggregation with one merge dataframe
            return pd.DataFrame(
                {
                    k: do_single_aggregation(
                        merged_df=merged_df,
                        values_col=values_col,
                        origin_node_id_col=origin_node_id_col,
                        decayed_weights_col=decayed_weights_col,
                        aggregation=v,
                    )
                    for k, v in aggregation.items()
                }
            )
        else:
            return do_single_aggregation(
                merged_df=merged_df,
                values_col=values_col,
                origin_node_id_col=origin_node_id_col,
                decayed_weights_col=decayed_weights_col,
                aggregation=aggregation,
            )

    def write(self, edges_filename: str, nodes_filename: str):
        """
        Write this object to 2 geoparquet files
        """
        self.nodes.to_parquet(nodes_filename)
        self.edges.to_parquet(edges_filename)

    @staticmethod
    def read(
        edges_filename: str,
        nodes_filename: str,
        from_nodes_col: str = "u",
        to_nodes_col: str = "v",
        edge_costs_col: str = "length",
    ):
        """
        Read a PandanaNetwork from 2 parquet files
        """
        return PandanaNetwork(
            edges=gpd.read_parquet(edges_filename),
            nodes=gpd.read_parquet(nodes_filename),
            from_nodes_col=from_nodes_col,
            to_nodes_col=to_nodes_col,
            edge_costs_col=edge_costs_col,
        )

    @staticmethod
    def from_osmnx_local_streets_place_query(place_query: str):
        """
        Use osmnx to grab local street network using settings appropriate for this library
        """
        osmnx.settings.bidirectional_network_types = ["drive"]
        graph = osmnx.graph_from_place(
            place_query,
            network_type="drive",
            custom_filter='["highway"~"residential|secondary|tertiary|secondary_link|tertiary_link|unclassified"]',
        )
        nodes, edges = osmnx.graph_to_gdfs(graph)

        return PandanaNetwork(
            edges=edges.reset_index(level=2, drop=True)[["length", "geometry"]],
            nodes=nodes[["geometry"]],
        )
