import geopandas as gpd
import pandas as pd


def nearest_nodes(values_gdf: gpd.GeoDataFrame, nodes_gdf: gpd.GeoDataFrame):
    """
    Map each point in values_gdf to its nearest node in nodes_gdf
    :param values_gdf:
    :param nodes_gdf:
    :return:
    """
    return (
        values_gdf.to_crs(epsg=3857)
        .sjoin_nearest(nodes_gdf.to_crs(epsg=3857))
        .rename(columns={"index_right": "node_id"})
        .set_index("node_id")
    )


def aggregate(
    values_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    max_distance: float,
    value_col="value",
    aggregation="sum",
    decay="linear",
    # TODO how to pass custom aggregation?
):
    group_funcs = {
        ("sum", "linear"): lambda x: (x[value_col] * x["weight"] / max_distance).sum()
    }
    return (
        edges_df[edges_df.weight <= max_distance]
        .merge(values_df[[value_col]], how="inner", left_on="to", right_index=True)
        .groupby("from")
        .apply(group_funcs[(aggregation, decay)], include_groups=False)
    )
