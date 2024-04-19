import random

import geopandas as gpd
import pandas as pd
import shapely

num_samples = 2000
max_distance = 1000


gdf = gpd.read_parquet("oakland_nodes.geoparquet")
bounds = gdf.total_bounds
# make num_samples random points within total_bounds
gdf2 = gpd.GeoDataFrame(
    [{"value": random.uniform(0, 100)} for i in range(num_samples)],
    geometry=[
        shapely.Point(
            random.uniform(bounds[0], bounds[2]), random.uniform(bounds[1], bounds[3])
        )
        for _ in range(num_samples)
    ],
    crs=gdf.crs,
)

gdf.to_crs(epsg=3857, inplace=True)
print(gdf)
gdf2.to_crs(epsg=3857, inplace=True)
# map each random point to the nearest osm_id
gdf3 = (
    gdf2.sjoin_nearest(gdf)
    .rename(columns={"index_right": "osm_id"})
    .set_index("osm_id")
)
print(gdf3.index.value_counts())


df = pd.read_parquet("oakland_edges.parquet")
print(df)
# assign the value (multiple times) to the edges dataframe
# how=inner means we drop nodes that don't have a value assigned, for performance
df = df[df.weight <= max_distance].merge(
    gdf3[["value"]], how="inner", left_on="to", right_index=True
)
print(df)


def group_func(x):
    # TODO support other decay functions
    linear_decay = x["weight"] / max_distance
    return (x["value"] * linear_decay).sum()


aggregated_values = df.groupby("from").apply(group_func, include_groups=False)
print(aggregated_values)
