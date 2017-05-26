import tensorflow as tf

from test import test_model
from train import train_model

# hide everything from tensorflow but errors
tf.logging.set_verbosity(tf.logging.ERROR)


def main():
    train_file = raw_input("Enter the training batch file path:\n")
    test_file = raw_input("Enter the testing batch file path:\n")
    output_file = raw_input("Enter the output file path:\n")
    classifier = train_model(train_file)
    test_model(classifier, test_file, output_file)
    print "Done!"


if __name__ == "__main__":
    main()
