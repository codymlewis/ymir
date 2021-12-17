"""
Collection of haiku implemented neural network models.  
Each of the classification models are functions take as
arguments the number of classes, the input tensor, and a flags stating whether to return the activation
`https://arxiv.org/abs/2004.03657 <https://arxiv.org/abs/2004.03657>`_ or
classification.
"""


import jax
import haiku as hk


def Logistic(classes, x, act=False):
    """LeNet 300-100 network from `https://doi.org/10.1109/5.726791 <https://doi.org/10.1109/5.726791>`_"""
    x = hk.Sequential([
        hk.Flatten(),
        hk.Linear(classes),
    ])(x)
    if act:
        return x
    return jax.nn.sigmoid(x)


def LeNet_300_100(classes, x, act=False):
    """LeNet 300-100 network from `https://doi.org/10.1109/5.726791 <https://doi.org/10.1109/5.726791>`_"""
    x = hk.Sequential([
        hk.Flatten(),
        hk.Linear(300), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
    ])(x)
    if act:
        return x
    return hk.Linear(classes)(x)


def LeNet(classes, x, act=False):
    """LeNet network from `https://doi.org/10.1109/5.726791 <https://doi.org/10.1109/5.726791>`_"""
    x = hk.Sequential([
        hk.Conv2D(20, kernel_shape=5, stride=1),
        hk.MaxPool(2, 2, "SAME"), jax.nn.relu,
        hk.Conv2D(50, kernel_shape=5, stride=1),
        hk.MaxPool(2, 2, "SAME"), jax.nn.relu,
        hk.Flatten(),
        hk.Linear(500),
    ])(x)
    if act:
        return x
    return hk.Linear(classes)(x)


def ConvLeNet(classes, x, act=False):
    """LeNet 300-100 network with a convolutional layer and max pooling layer prepended"""
    x = hk.Sequential([
        hk.Conv2D(64, kernel_shape=11, stride=4), jax.nn.relu,
        hk.MaxPool(3, strides=2, padding="VALID"),
        hk.Flatten(),
        hk.Linear(300), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
    ])(x)
    if act:
        return x
    return hk.Linear(classes)(x)


def CNN(classes, x, act=False):
    """A convolutional neural network"""
    x = hk.Sequential([
        hk.Conv2D(64, kernel_shape=11, stride=2), jax.nn.relu,
        hk.MaxPool(3, strides=2, padding="VALID"),
        hk.Conv2D(16, kernel_shape=5, stride=2), jax.nn.relu,
        hk.MaxPool(3, strides=2, padding="VALID"),
        hk.Flatten(),
        hk.Linear(500), jax.nn.relu,
    ])(x)
    if act:
        return x
    return hk.Linear(classes)(x)