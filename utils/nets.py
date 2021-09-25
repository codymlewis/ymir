import jax
from jax._src.numpy.lax_numpy import pad
import jax.numpy as jnp
import haiku as hk

class LeNet(hk.Module):
    def __init__(self, classes, name=None):
        super().__init__(name=name)
        self.layers = [
            hk.Flatten(),
            hk.Linear(300), jax.nn.relu,
            hk.Linear(100), jax.nn.relu,
            hk.Linear(classes)
        ]

    def __call__(self, x):
        return hk.Sequential(self.layers)(x)
    
    def act(self, x):
        return hk.Sequential(self.layers[:-1])(x)


class ConvLeNet(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.layers = [
            hk.Conv2D(64, kernel_shape=11, stride=4), jax.nn.relu,
            hk.MaxPool(3, strides=2, padding="VALID"),
            hk.Flatten(),
            hk.Linear(300), jax.nn.relu,
            hk.Linear(100), jax.nn.relu,
            hk.Linear(10)
        ]

    def __call__(self, x):
        return hk.Sequential(self.layers)(x)