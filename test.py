import tensorflow as tf
import numpy as np
import tenjin

def create_model(input_shape, output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='softmax')
    ])
    return model

X, y, train = tenjin.load('mnist')
model = create_model(input_shape=X[0].shape, output_shape=np.unique(y).shape[0])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
params = model.get_weights()
model.fit(X[train], y[train], validation_data=(X[~train], y[~train]), epochs=10, shuffle=True)