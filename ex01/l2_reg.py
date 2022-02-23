import sys
import numpy as np

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
    # Checking dim
    if theta.ndim != 2:
        s = "Unexpected dimension for theta."
        print(s, file=sys.stderr)
        return None
    # Checking data type, 'i': signed integer, 'u': unsigned integer,
    # 'f': float
    if theta.dtype.kind not in ['i', 'u', 'f']:
        s = "Unexpected data type for theta."
        print(s, file=sys.stderr)
        return None

    l2 = 0
    for t in theta[1:]:
        l2 += t ** 2
    return l2
    


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
    # Checking dim
    if theta.ndim != 2:
        s = "Unexpected dimension for theta."
        print(s, file=sys.stderr)
        return None
    # Checking data type, 'i': signed integer, 'u': unsigned integer,
    # 'f': float
    if theta.dtype.kind not in ['i', 'u', 'f']:
        s = "Unexpected data type for theta."
        print(s, file=sys.stderr)
        return None

    l2 = np.dot(theta[1:], theta[1:])
    return l2


if __name__ == '__main__':
    x = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
    # Example 1:
    iterative_l2(x)
    # Output:
    911.0
    # Example 2:
    l2(x)
    # Output:
    911.0
    y = np.array([[3],[0.5],[-6]])
    # Example 3:
    iterative_l2(y)
    # Output:
    36.25
    # Example 4:
    l2(y)
    # Output:
    36.25
