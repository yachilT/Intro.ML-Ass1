import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self, k, x_train: np.array, y_train: np.array):
        self.k = k
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, test): # gets a single test example and returns the predicted label
        distances = np.array([distance.euclidean(test, x) for x in self.x_train])
        indices = np.argsort(distances)
        k_indices = indices[:self.k]
        k_labels = self.y_train[k_indices].astype('int64')
        return np.argmax(np.bincount(k_labels))


def gensmallm(x_list: list, y_list: list, m: int):
    """
    gensmallm generates a random sample of size m along side its labels.

    :param x_list: a list of numpy arrays, one array for each one of the labels
    :param y_list: a list of the corresponding labels, in the same order as x_list
    :param m: the size of the sample
    :return: a tuple (X, y) where X contains the examples and y contains the labels
    """

    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x[:m], rearranged_y[:m]


# todo: complete the following functions, you may add auxiliary functions or define class to help you

def learnknn(k: int, x_train: np.array, y_train: np.array):
    """
    :param k: value of the nearest neighbour parameter k
    :param x_train: numpy array of size (m, d) containing the training sample
    :param y_train: numpy array of size (m, 1) containing the labels of the training sample
    :return: classifier data structure
    """
    return Classifier(k, x_train, y_train)

def predictknn(classifier, x_test: np.array):
    """

    :param classifier: data structure returned from the function learnknn
    :param x_test: numpy array of size (n, d) containing test examples that will be classified
    :return: numpy array of size (n, 1) classifying the examples in x_test
    """
    return np.array([classifier.predict(sample) for sample in x_test]).reshape(x_test.shape[0], 1)


def get_tests(x_list, y_list):
    assert len(x_list) == len(y_list), 'The length of x_list and y_list should be equal'

    x = np.vstack(x_list)
    y = np.concatenate([y_list[j] * np.ones(x_list[j].shape[0]) for j in range(len(y_list))])

    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    rearranged_x = x[indices]
    rearranged_y = y[indices]

    return rearranged_x, rearranged_y



def simple_test():
    data = np.load('mnist_all.npz')

    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']

    x_train, y_train = gensmallm([train0, train1, train2, train3], [0, 1, 2, 3], 100)


    x_test, y_test = gensmallm([test0, test1, test2, test3], [0, 1, 2, 3], 50)

    classifer = learnknn(5, x_train, y_train)

    preds = predictknn(classifer, x_test)
    # tests to make sure the output is of the intended class and shape
    assert isinstance(preds, np.ndarray), "The output of the function predictknn should be a numpy array"
    assert preds.shape[0] == x_test.shape[0] and preds.shape[1] == 1, f"The shape of the output should be ({x_test.shape[0]}, 1)"

    # get a random example from the test set
    i = np.random.randint(0, x_test.shape[0])

    # this line should print the classification of the i'th test sample.
    print(f"The {i}'th test sample was classified as {preds[i]}")


if __name__ == '__main__':

    # before submitting, make sure that the function simple_test runs without errors
    # simple_test()

    data = np.load('mnist_all.npz')
    digits = [2,3,5,6]
    avg_errors = np.array([])
    max_errors = np.array([])
    min_errors = np.array([])
    (x_test, y_test) = get_tests([data[f'test{k}'] for k in digits], digits)
    for i in range(1, 100):
        error = np.array([])
        for j in range(10):
            (X, Y) = gensmallm([data[f'train{k}'] for k in digits], digits, i)
            classifier = learnknn(1, X, Y)


            preds = predictknn(classifier, x_test)
            error = np.append(error, [np.mean(preds != y_test)])

        avg_errors = np.append(avg_errors, np.mean(error))
        max_errors = np.append(max_errors, np.max(error))
        min_errors = np.append(min_errors, np.min(error))
        print(i)

    plt.errorbar(range(1, 100), avg_errors, yerr = [np.array(avg_errors) - np.array(min_errors),
                                       np.array(max_errors) - np.array(avg_errors)], fmt = '-o')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Average Test Error')
    plt.title('Average Test Error vs. Training Sample Size')
    plt.show()






