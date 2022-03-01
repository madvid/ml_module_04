import numpy as np
import sys
from sigmoid import sigmoid_


def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of shape m * n.
        theta: has to be an numpy.array, a vector of shape (n + 1) * 1.
    Return:
        y_hat: a numpy.array of shape m * 1, when x and theta numpy arrays
            with expected and compatible shapes.
        None: otherwise.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if (not isinstance(x, np.ndarray)) \
                or (not isinstance(theta, np.ndarray)):
            s = "x or theta are not of the expected type (numpy array)."
            print(s, file=sys.stderr)
            return None
        if x.ndim != 2 or \
                (x.shape[1] + 1 != theta.shape[0]) \
                or (theta.shape[1] != 1):
            s = "x or theta not 2 dimensional array " \
                + "or mismatching shape between x and theta"
            print(s, file=sys.stderr)
            return None

        x_ = np.hstack((np.ones((x.shape[0], 1)), x))
        ypred = sigmoid_(np.dot(x_, theta))
        return ypred
    except:
        return None


if __name__ == "__main__":
    print("# Example 1")
    x = np.array([[4]])
    theta = np.array([[2], [0.5]])
    res = logistic_predict_(x, theta)
    # Output:
    expected = np.array([[0.98201379]])
    print("my log pred:".ljust(25), res.reshape(1, -1))
    print("expected log pred:".ljust(25), expected.reshape(1, -1))

    print("\n# Example 1")
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    res = logistic_predict_(x2, theta2)
    # Output:
    expected = np.array([[0.98201379],
                         [0.99624161],
                         [0.97340301],
                         [0.99875204],
                         [0.90720705]])
    print("my log pred:".ljust(25), res.reshape(1, -1))
    print("expected log pred:".ljust(25), expected.reshape(1, -1))

    print("\n# Example 2")
    x3 = np.array([[0, 2, 3, 4],
                   [2, 4, 5, 5],
                   [1, 3, 2, 7]])
    theta3 = np.array([[-2.4],
                       [-1.5],
                       [0.3],
                       [-1.4],
                       [0.7]])
    res = logistic_predict_(x3, theta3)
    # Output:
    expected = np.array([[0.03916572],
                         [0.00045262],
                         [0.2890505]])
    print("my log pred:".ljust(25), res.reshape(1, -1))
    print("expected log pred:".ljust(25), expected.reshape(1, -1))
