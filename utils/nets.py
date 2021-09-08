import jax
import jax.numpy as jnp
import haiku as hk

def lenet(get_acts=False):
    def net_fn(batch):
        """Standard LeNet-300-100 MLP network."""
        x = batch["image"].astype(jnp.float32) / 255.
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(300), jax.nn.relu,
            hk.Linear(100), jax.nn.relu,
            hk.Linear(10)
        ])
        return mlp(x)

    def net_act(batch):
        x = batch["image"].astype(jnp.float32) / 255.
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(300), jax.nn.relu,
            hk.Linear(100), jax.nn.relu,
        ])
        return mlp(x)
    if get_acts:
        return hk.without_apply_rng(hk.transform(net_fn)), hk.without_apply_rng(hk.transform(net_act))
    return hk.without_apply_rng(hk.transform(net_fn))