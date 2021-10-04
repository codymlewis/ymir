# Garrison
Functionality specific to the global server of the FL system

## Design
The design of a resulting program is ultimately subjective to the FL algorithm being applied,
thus we recommend using the higher level API for most applications.

In the low-level API typically consists of a Server class for the storage of persistent information,
an update function for updating that information, and a scale function to scale the client gradients.
Afterwards there are general functions used to apply the scale to each of the client gradients, `ymir.garrison.apply_scale(scale, all_grads)`, and to
update the global model, `ymir.garrison.update(opt)(params, opt_state, ymir.garrison.sum_grads(all_grads))`.

The following snippet demonstrates a generic example using the low-level API manually
~~~python
server_update = ymir.garrison.update(opt)

for round in range(N):
    for client in clients:
        grads, client.opt_state = client_update(params, client.opt_state, *next(client.data))
        all_grads.append(grads)

    server.histories = alg.update(server.histories, all_grads)
    alpha = alg.scale(server, all_grads)
    all_grads = ymir.garrison.apply_scale(alpha, all_grads)

    params, opt_state = server_update(params, opt_state, ymir.garrison.sum_grads(all_grads))
~~~