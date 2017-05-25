import tensorflow as tf
import numpy as np

FEATURES = range(0, 14)
LABEL = -1


def load_batch(filename):
    return np.genfromtxt(filename,
                         dtype=np.int,
                         skip_header=1,
                         delimiter=',',
                         missing_values='?',
                         filling_values=-1)


def pipeline_from_file(filename, with_labels=True):
    raw = load_batch(filename)
    feature_columns = {}
    for col in FEATURES:
        column = raw[:, col]
        tensor = tf.constant(column, shape=[column.size, 1])
        feature_columns[str(col)] = tensor
    if with_labels:
        labels = tf.constant(raw[:, LABEL])
    else:
        labels = None
    return feature_columns, labels
