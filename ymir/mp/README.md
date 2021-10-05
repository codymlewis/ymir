# MP
Shared utilities to be used between both the endpoints and the server

## Datasets
Contains the `Dataset` class which is constructed by numpy-based datasets divided into two collections, X and y, where X is the collection of samples and y is the
collection of labels. The `Dataset` class contains the abstract methods `.train()` and `.test()` which return the train and test subsets of the data in the 
form `(X, y)`. It also contains the method `.get_iter(split, batch_size=None, filter=None, map=None)` which returns an iterator of the split specified subset
of the data, if specified the iterator an yeild with size `batch_size`, additionally the `filter` and `map` arguments optionally perform preprocessing. The final
method is the abstract `.fed_split(batch_sizes, iid)` which splits the data into a list of data iterators based on an input list of batch sizes, the dataset split
can be specified as i.i.d.

## Losses
Collection of jax-based loss functions, they have a curried format so the network may be saved for a jit compiled function

## Metrics
Collection of functions for measuring the performance of the federated learning

## Nets
Collection of haiku defined neural networks
