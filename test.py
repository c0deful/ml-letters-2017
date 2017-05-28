from data import load_batch, pipeline


def test_model(classifier, test_filepath, output_filepath):
    test_data = load_batch(test_filepath, with_labels=False)
    print 'Testing on batch of size {}'.format(test_data.shape[0])
    predictions = classifier.predict_classes(input_fn=lambda: pipeline(test_data))
    output = open(output_filepath, 'w')
    for label in predictions:
        output.write(chr(label + ord('A')) + '\n')
    output.close()
