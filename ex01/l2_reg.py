import sys
import numpy as np


# ########################################################################## #
#                             Function definitions                           #
# ########################################################################## #
def show_(my_res, expected_res):
    """ Displays the calculated result and the expected result.
    Convenient for project correction consideration at 42.
    """
    print("My regularized l2 term:".ljust(35), my_res)
    print("Expected regularized l2 term:".ljust(35), expected_res)


def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.array, with a
    for-loop.
    Args:
        theta: has to be a numpy.array, a vector of shape n’ * 1.
    Return:
        The L2 regularization as a float.
        None if theta in an empty numpy.array.
        None if theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    # Checking type
    if not isinstance(theta, np.ndarray):
        s = "Unexpected type for theta."
        print(s, file=sys.stderr)
        return None
    # Checking dim and shape
    if (theta.ndim != 2) or (theta.shape[1] != 1):
        s = "Unexpected dimension/shape for theta."
        print(s, file=sys.stderr)
        return None
    # Checking data type, 'i': signed integer, 'u': unsigned integer,
    # 'f': float
    if theta.dtype.kind not in ["i", "u", "f"]:
        s = "Unexpected data type for theta."
        print(s, file=sys.stderr)
        return None

    l2 = 0
    for t in theta[1:]:
        l2 += t**2
    return np.squeeze(l2.astype(float))


def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.array, without any
    for-loop.
    Args:
        theta: has to be a numpy.array, a vector of shape n’ * 1.
    Return:
        The L2 regularization as a float.
        None if theta in an empty numpy.array.
        None if theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    # Checking type
    if not isinstance(theta, np.ndarray):
        s = "Unexpected type for theta."
        print(s, file=sys.stderr)
        return None
    # Checking dim and shape
    if (theta.ndim != 2) or (theta.shape[1] != 1):
        s = "Unexpected dimension/shape for theta."
        print(s, file=sys.stderr)
        return None
    # Checking data type, 'i': signed integer, 'u': unsigned integer,
    # 'f': float
    if theta.dtype.kind not in ["i", "u", "f"]:
        s = "Unexpected data type for theta."
        print(s, file=sys.stderr)
        return None

    l2 = np.dot(theta[1:].T, theta[1:])
    return np.squeeze(l2.astype(float))


# ########################################################################## #
#                                      Main                                  #
# ########################################################################## #
if __name__ == "__main__":
    x = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

    print("# Example 1:")
    my_res = iterative_l2(x)
    expected_res = 911.0
    show_(my_res, expected_res)

    print("# Example 2:")
    my_res = l2(x)
    expected_res = 911.0
    show_(my_res, expected_res)

    y = np.array([[3], [0.5], [-6]])
    print("# Example 3:")
    my_res = iterative_l2(y)
    expected_res = 36.25
    show_(my_res, expected_res)

    print("# Example 4:")
    my_res = l2(y)
    expected_res = 36.25
    show_(my_res, expected_res)
