# Scout
Endpoint side functionalities. `__init__.py` encompasses the basic endpoint functionality,
while there additionally exist other specialized modules.

The following snippet performs some local training for an endpoint:
~~~python
client_update = ymir.scout.update(opt, ymir.mp.losses.cross_entropy_loss(net, DATASET.classes))
client = ymir.scout.Client(opt_state, data, 8) 
grads, client.opt_state = client_update(params, client.opt_state, *next(client.data))
~~~