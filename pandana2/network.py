import geopandas as gpd


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
    print(values_gdf.to_crs(epsg=3857).sjoin_nearest(nodes_gdf.to_crs(epsg=3857)))
    return (
        values_gdf.to_crs(epsg=3857)
        .sjoin_nearest(nodes_gdf.to_crs(epsg=3857))
        .rename(columns={"index_right": "node_id"})
        .set_index("node_id")
    )
