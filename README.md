# pandana2

### Motivating use case

The canonical use case of this library is to perform a mean of home prices in cities using local street networks.  This solves two problems:

1. We use network distance instead of "crows flies" distance because home prices on opposite sides of a freeway, for instance, are not as related as homes on a locally connected street network.
2. Smoothly overlapping distances along the street network reduces discontinuities by not aggregating within polygons

In short, the goal is to compute a weighted average of all the observations within a certain distance (say 1500 meters) from every origin street intersection along the street network.  Weighted averages are easily computed so that observations closer to the origin node can be weighted higher.

### Relationship to 2015 pandana

The original pandana was written during my graduate research with Paul Waddell at UC Berkeley.  It leverages a huge C++ library for Contraction Hierachies which I eventually realized was useful to the routing queries but not useful to the aggregation queries.

10 years later, I decided to rewrite the library to not use C++ and only do the aggregations.  The "get nodes with X meters of each origin node in the network" part is written in numba.  The rest is vanilla pandas.

This should greatly ease extension and maintenance of the library since it's less than 1000 lines of code so far and much easier to deploy since it doesn't have to be compiled.

This version has feature parity with the 2015 version for the aggregations, doing sum, mean, median, standard deviation, min, and max with no decay, linear decay and exponential decay.

This version has weighted mean, weighted median, and weighted standard deviation which the previous version did not have.  Additional aggregations and decays should be quite trivial to add.

