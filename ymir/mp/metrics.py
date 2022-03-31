"""
Measure performance during experiments, and enable their recording at the end.
"""

import numpy as np
import sklearn.metrics as skm


def accuracy(model, data):
    """
    Get the accuracy score of the model on the data.
    """
    X, y = next(data)
    return skm.accuracy_score(y, np.argmax(model.predict(X), axis=1))
