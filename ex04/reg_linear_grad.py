import sys
import numpy as np


# ########################################################################## #
#                             Function definitions                           #
# ########################################################################## #
def show_(my_res, expected_res):
    """ Displays the calculated result and the expected result.
    Convenient for project correction consideration at 42.
    """
    if my_res is None:
        print("My reg linear gradient:".ljust(25), None)
    else:
        print("My reg linear gradient:".ljust(25), my_res.reshape(1,-1))
    if expected_res is None:
        print("Expected reg linear grad:".ljust(25), None)
    else:
        print("Expected reg linear grad:".ljust(25), expected_res.reshape(1,-1))


def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.array,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        x: has to be a numpy.array, a matrix of dimesion m * n.
        theta: has to be a numpy.array, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.array, a vector of shape (n + 1) * 1, containing the results
            of the formula for all j.
        None if y, x, or theta are empty numpy.array.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # Checking the type of the parameters
        if (not isinstance(y, np.ndarray)) or (not isinstance(x, np.ndarray)) \
                or (not isinstance(theta, np.ndarray)) \
                or (not isinstance(lambda_, (float, int))):
            s = "Unexpected type for at least one of the method parameters."
            print(s, file=sys.stderr)
            sys.exit()
        # Checking the dimension of the parameters
        if (y.ndim != 2) or (x.ndim != 2) or (theta.ndim != 2):
            s = "Unexpected dimension at least for one of the np.array."
            print(s, file=sys.stderr)
            sys.exit()
        # Checking shape compatibility between np.array
        if (y.shape[0] != x.shape[0]) or (y.shape[1] != 1) \
                or (x.shape[1] + 1 != theta.shape[0]) or (theta.shape[1] != 1):
            s = "Incompatible shape between the np.array parameters."
            print(s, file=sys.stderr)
            sys.exit()
        # Checking data type, 'i': signed integer, 'u': unsigned integer,
        # 'f': float
        if x.dtype.kind not in ["i", "u", "f"] \
                or y.dtype.kind not in ["i", "u", "f"] \
                or theta.dtype.kind not in ["i", "u", "f"]:
            s = "Unexpected data type for x or y or theta."
            print(s, file=sys.stderr)
            return None
        # Checking sign of lambdda_
        if (lambda_ < 0):
            s = "Notice: regularization coefficient is expected to be positive." \
                + " You hopefully know what you are doing."
            print(s, file=sys.stderr)
        
        reg_grad = np.zeros(theta.shape)
        for y_ii, x_ii in zip(y, x):
            pred_ = x_ii @ theta[1:] + theta[0]
            reg_grad[0] += pred_ - y_ii
            reg_grad[1:] += (x_ii * (pred_ - y_ii)).reshape(-1, 1)
        reg_grad[1:] += lambda_ * theta[1:]
        return (reg_grad) / y.shape[0]
    except:
        return None


def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.array,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        x: has to be a numpy.array, a matrix of dimesion m * n.
        theta: has to be a numpy.array, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.array, a vector of shape (n + 1) * 1, containing the results of
            the formula for all j.
        None if y, x, or theta are empty numpy.array.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # Checking the type of the parameters
        if (not isinstance(y, np.ndarray)) or (not isinstance(x, np.ndarray)) \
                or (not isinstance(theta, np.ndarray)) \
                or (not isinstance(lambda_, (float, int))):
            s = "Unexxpected type for at least one of the method parameters."
            print(s, file=sys.stderr)
            sys.exit()
        # Checking the dimension of the parameters
        if (y.ndim != 2) or (x.ndim != 2) or (theta.ndim != 2):
            s = "Unexpected dimension at least for one of the np.array."
            print(s, file=sys.stderr)
            sys.exit()
        # Checking shape compatibility between np.array
        if (y.shape[0] != x.shape[0]) or (y.shape[1] != 1) \
                or (x.shape[1] + 1 != theta.shape[0]) or (theta.shape[1] != 1):
            s = "Incompatible shape between the np.array parameters."
            print(s, file=sys.stderr)
            sys.exit()
        # Checking sign of lambdda_
        if (lambda_ < 0):
            s = "Notice: regularization coefficient is expected to be positive." \
                + " You hopefully know what you are doing."
            print(s, file=sys.stderr)

        x_ = np.hstack((np.ones((x.shape[0], 1)), x))
        theta_ = np.copy(theta)
        theta_[0] = 0
        reg_grad = x_.T @ ((x_ @ theta) - y) + lambda_ * theta_
        return reg_grad / y.shape[0]
    except:
        return None


# ########################################################################## #
#                                      Main                                  #
# ########################################################################## #
if __name__ == '__main__':
    x = np.array([[ -6, -7, -9],
                  [ 13, -2, 14],
                  [ -7, 14, -1],
                  [ -8, -4, 6],
                  [ -5, -9, 6],
                  [ 1, -5, 11],
                  [ 9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])

    print("# Example 1:")
    my_res = reg_linear_grad(y, x, theta, 1)
    expected_res = np.array([[-60.99],
                             [-195.64714286],
                             [863.46571429],
                             [-644.52142857]])
    show_(my_res, expected_res)

    print("\n# Example 2:")
    my_res = vec_reg_linear_grad(y, x, theta, 1)
    expected_res = np.array([[-60.99],
                             [-195.64714286],
                             [863.46571429],
                             [-644.52142857]])
    show_(my_res, expected_res)

    print("\n# Example 3:")
    my_res = reg_linear_grad(y, x, theta, 0.5)
    expected_res = np.array([[-60.99],
                             [-195.86142857],
                             [862.71571429],
                             [-644.09285714]])
    show_(my_res, expected_res)

    print("\n# Example 4:")
    my_res = vec_reg_linear_grad(y, x, theta, 0.5)
    expected_res = np.array([[-60.99 ],
                             [-195.86142857],
                             [862.71571429],
                             [-644.09285714]])
    show_(my_res, expected_res)

    print("\n# Example 5:")
    my_res = reg_linear_grad(y, x, theta, 0.0)
    expected_res = np.array([[-60.99],
                             [-196.07571429],
                             [861.96571429],
                             [-643.66428571]])
    show_(my_res, expected_res)

    print("\n# Example 6:")
    my_res = vec_reg_linear_grad(y, x, theta, 0.0)
    expected_res = np.array([[-60.99 ],
                             [-196.07571429],
                             [861.96571429],
                             [-643.66428571]])
    show_(my_res, expected_res)
