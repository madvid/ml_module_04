import numpy as np
import sys


inf_lim = -600  # To avoid RuntimeWarning do to overflow
sup_lim = 256  # To avoid RuntimeWarning do to overflow (even if it occurs to ~ [1e17 ; 1e18])

def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be an numpy.array, a vector
    Return:
        The sigmoid value as a numpy.array.
        None otherwise.
    Raises:
        This function should not raise any Exception.
    """
    try:
        x_ = np.array(x, copy=True)
        x_[x_ < inf_lim] = inf_lim
        x_[x_ > sup_lim] = sup_lim
        return np.divide(1., 1. + np.exp(-x))
    except:
        return None


if __name__ == "__main__":
    print("# Example 1:")
    x = np.array(-4)
    res = sigmoid_(x)
    # Output:
    expected = np.array([[0.01798620996209156]])
    print("Calculated sigmoid:".ljust(25), res.reshape(1, -1))
    print("Expected result:".ljust(25), expected.reshape(1, -1))

    print("\n# Example 2:")
    x = np.array(2)
    res = sigmoid_(x)
    # Output:
    expected = np.array([[0.8807970779778823]])
    print("Calculated sigmoid:".ljust(25), res.reshape(1, -1))
    print("Expected result:".ljust(25), expected.reshape(1, -1))

    print("\n# Example 3:")
    x = np.array([[-4], [2], [0]])
    res = sigmoid_(x)
    # Output:
    expected = np.array([[0.01798620996209156], [0.8807970779778823], [0.5]])
    print("Calculated sigmoid:".ljust(25), res.reshape(1, -1))
    print("Expected result:".ljust(25), expected.reshape(1, -1))
