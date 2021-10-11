# Ymir
JAX-based Federated learning library + repository of my FL research works

## Installation
As prerequisite, the `jax` and `jaxlib` libraries must be installed, we omit them from the
included `requirements.txt` as the installed library is respective to the system used. We direct
to first follow https://github.com/google/jax#installation then proceed with this section.

### The Ymir library
First make sure to install wheel
~~~sh
pip install wheel
~~~

after cloning the repository and opening a terminal inside the root folder, install the requirements with
~~~sh
pip install -r requirements.txt
~~~

and finally make and install the library with
~~~sh
make
~~~

#### Quick install
~~~sh
git clone git@github.com:codymlewis/ymir.git && cd ymir && pip install -r requirements.txt && make
~~~

## Usage
We provide examples of the library's usage in the `samples` folder. Though, generally
a program involves initializing shared values, then initialization of our `Coordinate`
object, and finally calling fit from that object.

The following is a generic example snippet
~~~python
# setup
dataset = ymir.mp.datasets.DATASET()
data = dataset.fed_split(batch_sizes, iid)
train_eval = dataset.get_iter("train", 10_000)
test_eval = dataset.get_iter("test")

net = hk.without_apply_rng(hk.transform(lambda x: ymir.mp.nets.Net(dataset.classes)(x)))
opt = optax.sgd(0.01)
params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])

model = ymir.Coordinate(
    alg_name, opt, params, ymir.mp.losses.cross_entropy_loss(net, dataset.classes), data
)

# Train/eval loop.
for round in range(total_epochs):
    model.fit()
~~~
