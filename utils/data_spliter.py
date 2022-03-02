import numpy as np

def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training
    and a test set, while respecting the given proportion of examples to
    be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of shape m * n.
        y: has to be an numpy.array, a vector of shape m * 1.
        proportion: has to be a float, the proportion of the
        dataset that will be assigned to the training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible shapes.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if (not isinstance(x, np.ndarray)) \
                or (not isinstance(y, np.ndarray)):
            return None

        if (x.shape[0] != y.shape[0]) \
                or (y.ndim != 2) or (x.ndim != 2) \
                or (y.shape[1] != 1):
            return None

        data = np.hstack((x, y))
        p = int(np.floor(x.shape[0] * proportion))
        np.random.default_rng(0).shuffle(data)

        x_train, y_train = data[:p, :-1], data[:p, -1:]
        x_test, y_test = data[p:, :-1], data[p:, -1:]

        return (x_train, x_test, y_train, y_test)
    except:
        return None


if __name__ == "__main__":
    x1 = np.array([[1], [42], [300], [10], [59]])
    y = np.array([[0], [1], [0], [1], [0]])

    print("# Example 0:")
    print(data_spliter(x1, y, 0.8))
    # Output:
    # (array([[ 10],
    #         [ 42],
    #         [  1],
    #         [300]]),
    #  array([[59]]),
    #  array([[1],
    #         [1],
    #         [0],
    #         [0]]),
    #  array([[0]]))

    print("\n# Example 1:")
    print(data_spliter(x1, y, 0.5))
    # Output:
    # (array([[42],
    #        [10]]),
    #  array([[ 59],
    #        [300],
    #        [ 1]]),
    #   array([[1],
    #         [1]]),
    #   array([[0],
    #         [0],
    #         [0]]))
    x2 = np.array([[1, 42],
                   [300, 10],
                   [59, 1],
                   [300, 59],
                   [10, 42]])
    y = np.array([[0], [1], [0], [1], [0]])

    print("\n# Example 2:")
    print(data_spliter(x2, y, 0.8))
    # Output:
    # (array([[10, 42],
    #         [59,  1],
    #         [ 1, 42],
    #         [300, 10]]),
    #  array([[300, 59]]),
    #  array([[0],
    #         [0],
    #         [0],
    #         [1]]),
    #  array([[1]]))

    print("\n# Example 3:")
    print(data_spliter(x2, y, 0.5))
    # Output:
    # (array([[300, 10],
    #         [  1, 42]]),
    #  array([[ 10, 42],
    #         [300, 59],
    #         [ 59,  1]]),
    #  array([[1],
    #         [0]]),
    #  array([[0],
    #         [1],
    #         [0]]))
