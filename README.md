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

### The data
Run the `dlall.sh` shell script to download, preprocess and save the datasets used.

## Usage
We provide examples of the library's usage in the `samples` folder. Though, generally
a program involves initializing shared values and the network architecture, then initialization
of our `Coordinate` object, and finally calling step from that object.

The following is a generic example snippet
~~~python
# setup
dataset = ymir.mp.datasets.load(DATASET)
data = dataset.fed_split(batch_sizes, DIST_LIST)
train_eval = dataset.get_iter("train", 10_000)
test_eval = dataset.get_iter("test")

net = hk.without_apply_rng(hk.transform(lambda x: ymir.mp.nets.Net(dataset.classes)(x)))
opt = optax.sgd(0.01)
params = net.init(jax.random.PRNGKey(42), next(test_eval)[0])
opt_state = opt.init(params)
loss = ymir.mp.losses.cross_entropy_loss(net, dataset.classes)
network = ymir.mp.network.Network(opt, loss)
network.add_controller("main", is_server=True)
for d in data:
    network.add_host("main", ymir.scout.Collaborator(opt_state, d, CLIENT_EPOCHS))

model = ymir.Coordinate(AGG_ALG, opt, opt_state, params, network)

# Train/eval loop.
for round in range(TOTAL_EPOCHS):
    model.step()
~~~
