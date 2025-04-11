# pandana2

An example aggregating home values in Oakland.  This is a screenshot - click [here](https://nbviewer.org/urls/gist.githubusercontent.com/fscottfoti/6dbdd18c3a065d5517dfd390004f1cf1/raw/21919407619889b984fd36201f1b089ac46e0f81/gistfile1.txt) and scroll to the bottom for an interative map.

<img width="864" alt="Screenshot 2025-04-11 at 4 48 27 PM" src="https://github.com/user-attachments/assets/6bb7dc24-db92-4565-8eab-07c2e1edbd0a" />

### Motivating use case

The motivating use case of this library is to perform a mean of home prices (or anything else) in cities using local street networks.  This solves two problems:

1. We use network distance instead of "crows flies" distance because home prices on opposite sides of a freeway, for instance, are not as related as homes on a locally connected street network.
2. Using smoothly overlapping distances along the street network reduces discontinuities that occur when observations are aggregated into polygons.

In short, the goal of this library is to compute weighted aggregations of all the observations within a certain network distance (e.g. 1500 meters or so) from every origin street intersection along the local street network.  Weighted averages are easily computed so that observations closer to the origin node can be weighted higher.

### Relationship to 2015 pandana

The original pandana was written during my graduate research with Paul Waddell at UC Berkeley.  It leverages a huge C++ library for Contraction Hierachies which I eventually realized was useful to the routing queries but not useful to the aggregation queries.  10 years later, I finally got around to rewriting the library to not use C++ and only do the aggregations.

The "get nodes within X meters of each origin node in the network" part is written in numba.  The rest is vanilla pandas.  This should greatly ease extension and maintenance of the library since it's less than 1000 lines of code so far and much easier to deploy since it doesn't have to be compiled.

This version has feature parity with the 2015 version for the aggregations: it supports sum, mean, median, standard deviation, min, and max with no decay, linear decay and exponential decay.  This version has weighted mean, weighted median, and weighted standard deviation which the previous version did not have.  Additional aggregations and decays should be quite easy to add.

### Example Notebook

The example Notebook is the best place to start to use this library.  View the very short example Notebook [here](https://nbviewer.org/urls/gist.githubusercontent.com/fscottfoti/6dbdd18c3a065d5517dfd390004f1cf1/raw/21919407619889b984fd36201f1b089ac46e0f81/gistfile1.txt).  Note the interactive map at the bottom of the notebook.

### Documentation

There's just not a ton of documentation requred.  Follow the example to see how to get a network and how to map data to the network.  Otherwise, the only method that matters is:

```
def aggregate(
  values: pd.Series,
  decay_func: PandanaDecayFunction,
  aggregation: Aggregation | dict[str, Aggregation],
) -> pd.Series | pd.DataFrame:
```

`values` is a series where the index is node_ids from the nodes of the network and the values are floating point values you want to aggregate.  In other words, it's the values and the node_ids they are located at (use the `nearest_nodes` method to assign node ids).  node_ids can and likely will be repeated in the index (i.e. not unique).

`decay_func` is either pandana2.NoDecay, pandana2.LinearDecay or pandana2.ExponentialDecay which defines how to map a distance along the network to a weight to be used in one of the aggregations.  The idea is that observations futher away should matter less when doing each aggregation, or you can use NoDecay to weight them equallly. 

`aggregation` is usually a string like "sum", "mean", "min", "max", "median", "std" (standard deviation).  All of the aggregations can be weighted except min and max which will ignore the weights.  You can pass a dict of aggregations to compute more than one aggregation for the same input Series.

The method will return a Series which is indexed the same as the nodes on the network.  NaN will be returned if there are no observations within the distance requested.  A DataFrame will be returned in the case the aggregation parameter is a dictionary.
