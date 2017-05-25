import tensorflow as tf
from pipeline import pipeline_from_file, FEATURES

LAYERS = [10, 20, 10]


def train_model(filename):
    # TODO create names for the columns
    columns = [tf.contrib.layers.real_valued_column(str(col)) for col in FEATURES]
    classifier = tf.contrib.learn.DNNClassifier(
        feature_columns=columns,
        hidden_units=LAYERS,
    )

    # TODO actually train the classifier
    # classifier.fit(input_fn=lambda: pipeline_from_file(filename),
    #                steps=100)

    return classifier
