import sys
import numpy as np


def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model from two
    non-empty numpy.array, without any for loop. The two arrays must have
    the same shapes.
    Args:
        y: has to be an numpy.array, a vector of shape m * 1.
        y_hat: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a vector of shape n * 1.
        lambda_: has to be a float.
    Return:
        The regularized loss as a float.
        None if y, y_hat, or theta are empty numpy.array.
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
        if (y.shape[1] != 1) \
            or (y_hat.shape[1] != 1) \
                or (y_hat.shape[0] != y.shape[0]):
            s = "Shape issue: either y and/or y_hat are not 2 dimensional," \
                + " or not the same number of lines."
            print(s, file=sys.stderr)
            return None
        # Checking theta dimension
        if (not isinstance(theta, np.ndarray)) \
                or (theta.ndim != 2):
            s = "2-dimensional array is expected for theta parameter."
            print(s, file=sys.stderr)
            return None
        # Checking lambda_  parameter
        if not isinstance(lambda_, (int, float)):
            s = "Numeric type is expected for lambda_ parameter."
            print(s, file=sys.stderr)
            return None

        t_ = np.squeeze(theta[1:])
        loss = (y - y_hat).T @ (y - y_hat)
        reg = lambda_ * t_ @ t_
        return float(0.5 * (loss + reg) / y.shape[0])
    except:
        None

if __name__ == "__main__":
    y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
    y_hat = np.array([[3],[ 13],[ -11.5],[ 5],[ 11],[ 5],[ -20]])
    theta = np.array([[1],[ 2.5],[ 1.5],[ -0.9]])
    # Example 1:
    reg_loss_(y, y_hat, theta, .5)
    # Output:
    0.8503571428571429
    # Example 2:
    reg_loss_(y, y_hat, theta, .05)
    # Output:
    0.5511071428571429
    # Example 3:
    reg_loss_(y, y_hat, theta, .9)
    # Output:
    1.116357142857143