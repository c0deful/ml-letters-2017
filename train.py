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
    columns = [tf.contrib.layers.real_valued_column(str(col))
               for col in FEATURE_COLS]
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=columns,
        hidden_units=[200, 200],
        n_classes=N_LABELS,
        dropout=0.5)

    # load and split data
    print "Loading training data."
    data = load_batch(filename)
    overall_size = data.shape[0]
    validation_size = int(overall_size * validation_ratio)
    learn_size = overall_size - validation_size
    learn, validation = np.array_split(data, [learn_size])
    print "Finished loading data. Samples count = {}".format(overall_size)

    # learning
    print "Starting training using batch of size {}".format(learn_size)
    classifier.fit(input_fn=lambda: pipeline(learn),
                   steps=learn_size)
    accuracy = get_accuracy(classifier, learn)
    print "Accuracy on learning samples = {}".format(accuracy)

    # validation
    if validation_size > 0:
        print "Starting validation using batch of size {}".format(validation_size)
        accuracy = get_accuracy(classifier, validation)
        print "Accuracy on validation samples = {}".format(accuracy)

    return classifier

if __name__ == "__main__":
    train_model("data/letter-recognition-train.csv", 0.3)
