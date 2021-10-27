import jax
import optax

from typing import Any, Callable, NamedTuple, Optional, Union


"""
Unique optimizers proposed in the FL literature
"""


def pgd(learning_rate, mu):
    """
    Perturbed gradient descent proposed as the mechanism for FedProx in https://arxiv.org/abs/1812.06127
    """
    return optax.chain(
        add_reg(mu),
        optax.scale(learning_rate)
    )


def add_reg(
    mu: float
) -> optax.GradientTransformation:
  """
  """

  def init_fn(_):
    return optax.EmptyState()

  def update_fn(updates, state, params):
    if params is None:
      raise ValueError("params argument required for this transform")
    updates = jax.tree_multimap(lambda g, p: g + mu * ((p - g) - p), updates, params)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)