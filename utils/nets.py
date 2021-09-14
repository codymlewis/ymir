import jax
import jax.numpy as jnp
import haiku as hk

class LeNet(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.layers = [
            hk.Flatten(),
            hk.Linear(300), jax.nn.relu,
            hk.Linear(100), jax.nn.relu,
            hk.Linear(10)
        ]

    def __call__(self, batch):
        x = batch["image"].astype(jnp.float32) / 255.
        return hk.Sequential(self.layers)(x)
    
    def act(self, batch):
        x = batch["image"].astype(jnp.float32) / 255.
        return hk.Sequential(self.layers[:-1])(x)