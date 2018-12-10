import numpy as np


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
dict_metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}