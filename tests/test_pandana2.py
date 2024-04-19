import osmnx

from pandana2 import network, aggregations


# TODO full-scale test?  does it work for whole Bay Area?


def get_amenity_as_dataframe(place_query: str, amenity: str):
    restaurants = osmnx.features_from_place(
        place_query, {"amenity": amenity}
    ).reset_index()
    restaurants = restaurants[restaurants.element_type == "node"][["name", "geometry"]]
    restaurants["count"] = 1
    return restaurants


def test_workflow():
    place_query = "Orinda, CA"
    g = osmnx.graph_from_place(place_query)
    nodes, edges = network.nodes_and_edges(g, 500)

    restaurants_df = get_amenity_as_dataframe(place_query, "restaurant")
    restaurants_df = aggregations.nearest_nodes(restaurants_df, nodes)
    assert restaurants_df.index.isin(nodes.index).all()

    group_func = aggregations.linear_decay_aggregation(500, "count", "sum")
    aggregations_series = aggregations.aggregate(restaurants_df, edges, group_func)
    assert aggregations_series.index.isin(nodes.index).all()
    assert aggregations_series.min() >= 0
    assert aggregations_series.max() <= 8

    # TODO if you don't visualize it, it's not real
    # TODO setup.py, test workflow, gitignore, imports are all red
