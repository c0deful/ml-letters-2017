import numpy as np
import tensorflow as tf

from config import FEATURE_COLS, LABEL_COL


def load_batch(filename):
    return np.genfromtxt(filename,
                         dtype=np.int,
                         skip_header=1,
                         delimiter=',',
                         converters={LABEL_COL: lambda c: ord(c) - ord('A')},
                         missing_values='?',
                         filling_values=-1)


def pipeline(raw, with_labels=True):
    feature_columns = {}
    for col in FEATURE_COLS:
        column = raw[:, col]
        tensor = tf.constant(column, shape=[column.size, 1])
        feature_columns[str(col)] = tensor
    if with_labels:
        labels = tf.constant(raw[:, LABEL_COL])
    else:
        labels = None
    return feature_columns, labels
