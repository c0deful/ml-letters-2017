import numpy as np
import tensorflow as tf

from config import FEATURE_COLS, N_LABELS
from data import pipeline, load_batch


def get_accuracy(classifier, data):
    return classifier.evaluate(
        input_fn=lambda: pipeline(data),
        steps=data.shape[0])['accuracy']


def train_model(filename, validation_ratio=0.):
    # define model to be trained
    columns = [tf.contrib.layers.real_valued_column(str(col),
                                                    dtype=tf.int8)
               for col in FEATURE_COLS]
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=columns,
        hidden_units=[100, 100],
        n_classes=N_LABELS,
        dropout=0.3)

    # load and split data
    print 'Loading training data.'
    data = load_batch(filename)
    overall_size = data.shape[0]
    learn_size = int(overall_size * (1 - validation_ratio))
    learn, validation = np.array_split(data, [learn_size])
    print 'Finished loading data. Samples count = {}'.format(overall_size)

    # learning
    print 'Training using batch of size {}'.format(learn_size)
    classifier.fit(input_fn=lambda: pipeline(learn),
                   steps=learn_size)

    if validation_ratio > 0:
        validate_model(classifier, learn, validation)

    return classifier


def validate_model(classifier, learn_data, validation_data):
    accuracy = get_accuracy(classifier, learn_data)
    print 'Accuracy on learning samples = {}'.format(accuracy)

    print 'Validating using batch of size {}'.format(validation_data.shape[0])
    accuracy = get_accuracy(classifier, validation_data)
    print 'Accuracy on validation samples = {}'.format(accuracy)


if __name__ == '__main__':
    train_model('data/letter-recognition-train.csv', 0.3)
