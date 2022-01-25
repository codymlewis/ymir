# Ymir
[![Tests status](https://github.com/codymlewis/ymir/actions/workflows/main.yml/badge.svg)](https://github.com/codymlewis/ymir/actions/workflows/main.yml)
[![License](https://img.shields.io/github/license/codymlewis/ymir?color=blue)](LICENSE)
![Commit activity](https://img.shields.io/github/commit-activity/m/codymlewis/ymir?color=red)

JAX-based Federated learning library

## Installation

As prerequisite, the `jax` and `jaxlib` libraries must be installed, we omit them from the
included `requirements.txt` as the installed library is respective to the system used. We direct
to first follow https://github.com/google/jax#installation then proceed with this section.

Afterwards, you can either quickly install the package with
```sh
pip install git+https://github.com/codymlewis/ymir.git
```
or build it from source with
```sh
pip install -r requirements.txt
make
```

## Usage

We provide examples of the library's usage in the `samples` folder. Though, generally
a program involves initializing shared values and the network architecture, then initialization
of our `Captain` object, and finally calling step from that object.

The following is a generic example snippet
```python
# Setup the dataset
dataset = ymir.mp.datasets.Dataset(X, y, train)
data = dataset.fed_split(batch_sizes, DIST_LIST)
train_eval = dataset.get_iter("train", 10_000)
test_eval = dataset.get_iter("test")

# Setup the network
net = JAX_BASED_NET
opt = optax.sgd(0.01)
params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
opt_state = opt.init(params)
loss = ymir.mp.losses.cross_entropy_loss(net, dataset.classes)
network = ymir.mp.network.Network(opt, loss)
network.add_controller("main", is_server=True)
for d in data:
    network.add_host("main", ymir.regiment.Scout(opt_state, d, CLIENT_EPOCHS))

model = ymir.garrison.AGG_ALG.Captain(params, opt, opt_state, network)

# Train/eval loop.
for round in range(TOTAL_EPOCHS):
    model.step()
```
