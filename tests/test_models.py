from absl.testing import absltest
from absl.testing import parameterized

import chex
import haiku as hk
import jax
import jax.numpy as jnp

import ymir

class TestModels(parameterized.TestCase):
    @parameterized.named_parameters(
        [
            {"testcase_name": f"_{model=}", "model": model}
            for model in ["LeNet", "ConvLeNet"]
        ]
    )
    def test_conv_models(self, model):
        X = jnp.ones((4, 3, 4, 4), dtype=jnp.float32)
        net = hk.without_apply_rng(hk.transform(lambda x: getattr(ymir.mp.models, model)(2)(x)))
        params = net.init(jax.random.PRNGKey(42), X[0])
        chex.assert_type(net.apply(params, X), jnp.float32)
        chex.assert_shape(net.apply(params, X), (4, 2))

    @parameterized.named_parameters(
        [
            {"testcase_name": f"_{model=}", "model": model}
            for model in ["Logistic", "LeNet_300_100"]
        ]
    )
    def test_ff_models(self, model):
        X = jnp.ones((4, 4), dtype=jnp.float32)
        net = hk.without_apply_rng(hk.transform(lambda x: getattr(ymir.mp.models, model)(2)(x)))
        params = net.init(jax.random.PRNGKey(42), X[0])
        chex.assert_type(net.apply(params, X), jnp.float32)
        chex.assert_shape(net.apply(params, X), (4, 2))


if __name__ == '__main__':
    absltest.main()