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


def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.array,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        x: has to be a numpy.array, a matrix of dimesion m * n.
        theta: has to be a numpy.array, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.array, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.array.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    pass


def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.array,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        x: has to be a numpy.array, a matrix of dimesion m * n.
        theta: has to be a numpy.array, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.array, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.array.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    pass


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
    
    print("\n# Example 2:")
    my_res = vec_reg_linear_grad(y, x, theta, 1)
    expected_res = np.array([[-60.99],
                             [-195.64714286],
                             [863.46571429],
                             [-644.52142857]])
    
    print("\n# Example 3:")
    my_res = reg_linear_grad(y, x, theta, 0.5)
    expected_res = np.array([[-60.99],
                             [-195.86142857],
                             [862.71571429],
                             [-644.09285714]])
    
    print("\n# Example 4:")
    my_res = vec_reg_linear_grad(y, x, theta, 0.5)
    expected_res = np.array([[-60.99 ],
                             [-195.86142857],
                             [862.71571429],
                             [-644.09285714]])
    
    print("\n# Example 5:")
    my_res = reg_linear_grad(y, x, theta, 0.0)
    expected_res = np.array([[-60.99],
                             [-196.07571429],
                             [861.96571429],
                             [-643.66428571]])
    
    print("\n# Example 6:")
    my_res = vec_reg_linear_grad(y, x, theta, 0.0)
    expected_res = np.array([[-60.99 ],
                             [-196.07571429],
                             [861.96571429],
                             [-643.66428571]])
