import jax
import haiku as hk

"""
Neural networks to be used with FL
"""


class LeNet_300_100(hk.Module):
    """LeNet 300-100 network"""
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


class LeNet(hk.Module):
    """LeNet network"""
    def __init__(self, classes, name=None):
        super().__init__(name=name)
        self.layers = [
            hk.Conv2D(20, kernel_shape=5, stride=1),
            hk.MaxPool(2, 2, "SAME"), jax.nn.relu,
            hk.Conv2D(50, kernel_shape=5, stride=1),
            hk.MaxPool(2, 2, "SAME"), jax.nn.relu,
            hk.Flatten(),
            hk.Linear(500),
            hk.Linear(classes)
        ]

    def __call__(self, x):
        return hk.Sequential(self.layers)(x)
    
    def act(self, x):
        return hk.Sequential(self.layers[:-1])(x)


class ConvLeNet(hk.Module):
    """LeNet 300-100 network with a convolutional layer and max pooling layer prepended"""
    def __init__(self, classes, name=None):
        super().__init__(name=name)
        self.layers = [
            hk.Conv2D(64, kernel_shape=11, stride=4), jax.nn.relu,
            hk.MaxPool(3, strides=2, padding="VALID"),
            hk.Flatten(),
            hk.Linear(300), jax.nn.relu,
            hk.Linear(100), jax.nn.relu,
            hk.Linear(classes)
        ]

    def __call__(self, x):
        return hk.Sequential(self.layers)(x)