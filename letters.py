from train import train_model
from test import test_model


def main():
    # TODO make this interactive for final solution
    # train_filepath = raw_input("Enter the training batch file path:\n")
    # test_filepath = raw_input("Enter the testing batch file path:\n")
    # output_filepath = raw_input("Enter the output file path:\n")
    train_filepath = "data/letter-recognition-train.csv"
    test_filepath = "data/letter-recognition-test-without-decisions.csv"
    output_filepath = "out"
    classifier = train_model(train_filepath)
    test_model(classifier, test_filepath, output_filepath)
    print "Done!"


if __name__ == "__main__":
    main()
