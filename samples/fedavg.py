"""
Example of federated averaging on the MNIST dataset
"""

import tensorflow as tf
import numpy as np
import tenjin
from tqdm import trange

import ymir


def create_model(input_shape, output_shape, lr=0.1):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(300, activation="relu")(x)
    x = tf.keras.layers.Dense(100, activation="relu")(x)
    outputs = tf.keras.layers.Dense(output_shape, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'])
    return model


if __name__ == "__main__":
    print("Setting up the system...")
    num_clients = 10
    rng = np.random.default_rng(0)
    # Setup the dataset
    dataset = ymir.mp.datasets.Dataset(*tenjin.load('mnist'))
    batch_sizes = [32 for _ in range(num_clients)]
    data = dataset.fed_split(batch_sizes, ymir.mp.distributions.lda, rng)
    train_eval = dataset.get_iter("train", 10_000, rng=rng)
    test_eval = dataset.get_iter("test", 10_000, rng=rng)
    # Setup the network
    network = ymir.mp.network.Network()
    for d in data:
        network.add_client(ymir.regiment.Scout(create_model(dataset.input_shape, dataset.classes), d, 1, test_data=test_eval))
    learner = ymir.garrison.fedavg.Captain(create_model(dataset.input_shape, dataset.classes, lr=1), network, rng)
    print("Done, beginning training.")
    # Train/eval loop.
    for r in (pbar := trange(5000)):
        loss = learner.step()
        if r % 10 == 0:
            metrics = learner.model.test_on_batch(*next(test_eval), return_dict=True)
            pbar.set_postfix(metrics)
