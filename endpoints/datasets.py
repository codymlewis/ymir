import tensorflow_datasets as tfds

def load_dataset(split, batch_size, filter=None, map=None):
    """Loads the dataset as a generator of batches."""
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if filter is not None:
        ds = ds.filter(filter)
    if map is not None:
        ds = ds.map(map)
    ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))