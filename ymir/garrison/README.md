# Garrison
Functionality specific to the global server of the FL system

## Design
We use the term `Captain` to refer to the global server. It handles the collection of weights/gradients from
each Collaborator and the computation of the global update.

The following snippet demonstrates a generic example using the aggregators in module
~~~python
server = ymir.garrison.Captain(params, opt, opt_state, network)

for round in range(N):
    server.step()
~~~
