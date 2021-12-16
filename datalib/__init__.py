"""
.. include:: README.md
"""


from absl import logging

from datalib import cifar10
from datalib import kddcup99
from datalib import mnist

def download(dir, dataset):
    """Download a dataset to a directory."""
    if globals().get(dataset) is None:
        logging.error('Dataset %s not found', dataset)
        raise ValueError(f"Dataset {dataset} not found")
    globals()[dataset].download(f"{dir}/{dataset}")