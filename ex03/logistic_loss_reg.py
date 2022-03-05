import sys
import numpy as np


# ########################################################################## #
#                             Function definitions                           #
# ########################################################################## #
def show_(my_res, expected_res):
    """ Displays the calculated result and the expected result.
    Convenient for project correction consideration at 42.
    """
    print("My reg logistic l2 loss:".ljust(35), my_res)
    print("Expected reg logistic l2 loss:".ljust(35), expected_res)


def reg_log_loss_(y, y_hat, theta, lambda_, eps=1e-15):
    """Computes the regularized loss of a logistic regression model from two
    non-empty numpy.array, without any for loop. The two arrays must have
    the same shapes.
    Args:
        y: has to be an numpy.array, a vector of shape m * 1.
        y_hat: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a vector of shape n * 1.
        lambda_: has to be a float.
        eps: has to be a float, epsilon (default=1e-15).
    Return:
        The regularized loss as a float.
        None if y, y_hat, or theta is empty numpy.array.
        None if y or y_hat have component ouside [0 ; 1]
        None if y and y_hat do not share the same shapes.
        None if y or y_hat is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # Checking y is numpy array
        if (not isinstance(y, np.ndarray)) \
                or (not isinstance(y_hat, np.ndarray)):
            s = "Numpy arrays are expected."
            print(s, file=sys.stderr)
            return None

        # Checking the shape of y and y_hat
        if (y.ndim != 2) or (y_hat.ndim != 2) \
                or (y.shape[1] != 1) or (y_hat.shape[1] != 1) \
                or (y_hat.shape[0] != y.shape[0]):
            s = (
                "Shape issue: either y and/or y_hat are not 2 dimensional,"
                + " or not the same number of lines."
            )
            print(s, file=sys.stderr)
            return None
        # Checking theta dimension
        if (not isinstance(theta, np.ndarray)) or (theta.ndim != 2):
            s = "2-dimensional array is expected for theta parameter."
            print(s, file=sys.stderr)
            return None
        # Checking data type, 'i': signed integer, 'u': unsigned integer,
        # 'f': float
        if y.dtype.kind not in ["i", "u", "f"] \
                or y_hat.dtype.kind not in ["i", "u", "f"] \
                or theta.dtype.kind not in ["i", "u", "f"]:
            s = "Unexpected data type for y or y_hat or theta."
            print(s, file=sys.stderr)
            return None
        # Checking lambda_  parameter
        if not isinstance(lambda_, (int, float)):
            s = "Numeric type is expected for lambda_ parameter."
            print(s, file=sys.stderr)
            return None

        t_ = np.squeeze(theta[1:])
        y_ = np.squeeze(y)
        y_hat_ = np.squeeze(y_hat)
        loss = y_ @ np.log(y_hat_ + eps) + (1 - y_) @ np.log(1 - y_hat_ + eps)
        reg = lambda_ * t_ @ t_ / (2 * y.shape[0])
        return -loss / y.shape[0] + reg
    except:
        return None


# ########################################################################## #
#                                      Main                                  #
# ########################################################################## #
if __name__ == "__main__":
    y = np.array([[1], [1], [0], [0], [1], [1], [0]])
    y_hat = np.array([[.9], [0.79], [0.12], [0.04], [0.89], [0.93], [0.01]])
    theta = np.array([[1], [2.5], [1.5], [-0.9]])

    print("# Example 1:")
    my_res = reg_log_loss_(y, y_hat, theta, .5)
    expected_res = 0.43377043716475955
    show_(my_res, expected_res)

    print("\n# Example 2:")
    my_res = reg_log_loss_(y, y_hat, theta, .05)
    expected_res = 0.13452043716475953
    show_(my_res, expected_res)

    print("\n# Example 3:")
    my_res = reg_log_loss_(y, y_hat, theta, .9)
    expected_res = 0.6997704371647596
    show_(my_res, expected_res)
