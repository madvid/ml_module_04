import numpy as np
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from sigmoid import sigmoid_


# ########################################################################## #
#                             Function definitions                           #
# ########################################################################## #
def show_(my_res, expected_res):
    """ Displays the calculated result and the expected result.
    Convenient for project correction consideration at 42.
    """
    print("My reg log gradient:".ljust(25), my_res.reshape(1,-1))
    print("Expected reg log grad:".ljust(25), expected_res.reshape(1,-1))


def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.array,
    with two for-loops. The three arrays must have compatible shapes.
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
    # try:
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
    # Checking sign of lambdda_
    if (lambda_ < 0):
        s = "Notice: regularization coefficient is expected to be positive." \
            + " You hopefully know what you are doing."
        print(s, file=sys.stderr)
    
    reg_grad = np.zeros(theta.shape)
    for y_ii, x_ii in zip(y, x):
        pred_ = sigmoid_(np.dot(x_ii, theta[1:]) + theta[0])
        reg_grad[0] += pred_ - y_ii
        reg_grad[1:] += (x_ii * (pred_ - y_ii)).reshape(-1, 1)
    reg_grad[1:] += lambda_ * theta[1:]
    return (reg_grad) / y.shape[0]
    # except:
    #     return None


def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.array,
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
    # try:
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
    pred = sigmoid_(np.dot(x_, theta))
    theta_ = np.copy(theta)
    theta_[0] = 0
    reg_grad = x_.T @ (pred - y) + lambda_ * theta_
    return reg_grad / y.shape[0]
    # except:
    #     return None


# ########################################################################## #
#                                      Main                                  #
# ########################################################################## #
if __name__ == '__main__':
    x = np.array([[0, 2, 3, 4],
                  [2, 4, 5, 5],
                  [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    
    print("# Example 1.1:")
    my_res = reg_logistic_grad(y, x, theta, 1)
    expected_result = np.array([[-0.55711039],
                                [-1.40334809],
                                [-1.91756886],
                                [-2.56737958],
                                [-3.03924017]])
    show_(my_res, expected_result)

    print("\n# Example 1.2:")
    my_res = vec_reg_logistic_grad(y, x, theta, 1)
    expected_result = np.array([[-0.55711039],
                                [-1.40334809],
                                [-1.91756886],
                                [-2.56737958],
                                [-3.03924017]])
    show_(my_res, expected_result)

    print("\n# Example 2.1:")
    my_res = reg_logistic_grad(y, x, theta, 0.5)
    expected_result = np.array([[-0.55711039],
                                [-1.15334809],
                                [-1.96756886],
                                [-2.33404624],
                                [-3.15590684]])
    show_(my_res, expected_result)

    print("\n# Example 2.2:")
    my_res = vec_reg_logistic_grad(y, x, theta, 0.5)
    expected_resut = np.array([[-0.55711039],
                               [-1.15334809],
                               [-1.96756886],
                               [-2.33404624],
                               [-3.15590684]])
    show_(my_res, expected_result)

    print("\n# Example 3.1:")
    my_res = reg_logistic_grad(y, x, theta, 0.0)
    expected_result = np.array([[-0.55711039],
                                [-0.90334809],
                                [-2.01756886],
                                [-2.10071291],
                                [-3.27257351]])
    show_(my_res, expected_result)

    print("\n# Example 3.2:")
    my_res = vec_reg_logistic_grad(y, x, theta, 0.0)
    expected_result = np.array([[-0.55711039],
                                [-0.90334809],
                                [-2.01756886],
                                [-2.10071291],
                                [-3.27257351]])
    show_(my_res, expected_result)