import geopandas as gpd
import pandas as pd
from typing import Callable


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


def linear_decay_aggregation(max_distance: float, value_col: str, aggregation: str):
    return lambda x: (x[value_col] * x["weight"] / max_distance).agg(aggregation)


def aggregate(
    values_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    group_func: Callable[[pd.DataFrame], float],
) -> pd.Series:
    return (
        edges_df.merge(values_df, how="inner", left_on="to", right_index=True)
        .groupby("from")
        .apply(group_func, include_groups=False)
    )
