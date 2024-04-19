import osmnx

from pandana2 import network, aggregations


# TODO full-scale test?  does it work for whole Bay Area?

def test_workflow():
    g = osmnx.graph_from_place("Orinda, CA")
    nodes, edges = network.nodes_and_edges(g, 500)
    print(nodes)
    print(edges)

    restaurants = osmnx.features_from_place("Orinda, CA", {'amenity': 'restaurant'}).reset_index()
    restaurants = restaurants[restaurants.element_type == "node"][["name", "geometry"]]
    restaurants["count"] = 1
    print(restaurants)

    restaurants = aggregations.nearest_nodes(restaurants, nodes)
    print(restaurants)

    aggregations_df = aggregations.aggregate(restaurants, edges, 500, value_col="count")
    print(aggregations_df)

    # TODO if you don't visualize it, it's not real
