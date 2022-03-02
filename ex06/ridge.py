import numpy as np
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(1, path)
from mylinearregression import MyLinearRegression


# ########################################################################### #
#                                Constants                                    #
# ########################################################################### #
dct_attr = {'thetas': (np.ndarray),
            'alpha': (float),
            'max_iter': (int),
            'lambda_': (float)}


# ########################################################################### #
#                                Functions                                    #
# ########################################################################### #
def check_types(f):
    """
    """
    def inner(*args):
        _, ag1, ag2 = args
        if (not isinstance(ag1, np.ndarray)) \
                or (not isinstance(ag2, np.ndarray)):
            s = "Numpy arrays are expected."
            print(s, file=sys.stderr)
            sys.exit()
        return f(*args)
    return inner

def check_measurement(arg1, arg2):
    """
    """
    def decorator_check_measurement(f):
        """
        """
        def dim_shape(*args):
            """
            """
            myridge, arr_1, arr_2 = args
            if (arr_1.ndim != 2) or (arr_2.ndim != 2):
                s = "Dimension issue: 2-D arrays expected."
                print(s, file=sys.stderr)
                sys.exit()
            if (arg1 == 'x') and (arg2 == 'y'):
                if (arr_1.shape[0] != arr_2.shape[0]) \
                        or (arr_2.shape[1] != 1) \
                        or (myridge.thetas.shape[0] != arr_1.shape[1] + 1):
                    s = "Mismatch shapes: x must be (m * n) and y (m * 1)." \
                        + " Plus, (n + 1) should be equal the number " \
                        + "components of thetas in MyRidge instance."
                    print(s, file=sys.stderr)
                    sys.exit()
            if (arg1 == 'y') and (arg2 == 'y'):
                if (arr_1.shape[1] != 1) or (arr_2.shape[1] != 1) \
                        or (arr_1.shape[0] != arr_2.shape[0]):
                    s = "Shape issue: either y and/or y_hat are not 2 " \
                        + "dimensional, or not the same number of lines."
                    print(s, file=sys.stderr)
                    sys.exit()
            
            return f(*args)
        return dim_shape
    return decorator_check_measurement


class MyRidge(MyLinearRegression):
    
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
        super().__init__(thetas, alpha=alpha, max_iter=max_iter)

        # Checking the type of lambda_
        if not isinstance(lambda_, (int, float)):
            s = "Unexpected type for lambda_ parameter."
            print(s, file=sys.stderr)
            sys.exit()
        # Checking the sign of lambda_
        if lambda_ < 0:
            s = "Nagative value lambda_ parameter. " \
                + "I hope you know what you are doing."
            print(s, file=sys.stderr)
        self.lambda_ = lambda_


    def get_params_(self):
        """ Gets the attributes of the estimator.
        No particular output is expected in the subject, thus one might use
        __dict__.
        Args:
            No argument except the instance itself.
        Return:
            [dict]: dictionary containing all the attributes.
        """
        return self.__dict__


    def set_params_(self, new_val):
        """ Set new values to attributes.
        Args:
            new_val [dict]: new values for the attributes precised as keys.
        Return:
            None
        """
        try:
            for key, val in new_val.items():
                if key in ('thetas', 'max_iter', 'alpha', 'lambda_'):
                    if not isinstance(val, dct_attr[key]):
                        s = 'thetas, max_iter, alpha and lambda_ parameters are' \
                            + 'expected to be a specific type.'
                        print(s, file=sys.stderr)
                        return None
                setattr(self, key, val)
        except:
            s = "Something went wrong when using set_params."
            print(s, file=sys.stderr)
            return None


    def l2(self):
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
        l2 = np.dot(self.thetas[1:].T, self.thetas[1:])
        return np.squeeze(l2.astype(float))


    @check_types
    @check_measurement('y', 'y')
    def loss_(self, y, y_hat):
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
            loss = (y - y_hat).T @ (y - y_hat)
            reg = self.lambda_ * self.l2()
            return float(0.5 * (loss + reg) / y.shape[0])
        except:
            None

    @check_types
    @check_measurement('y', 'y')
    def loss_elem_(self, y, y_hat):
        """Computes the regularized element-wise loss of a linear regression
        model from two non-empty numpy.array, without any for loop. The two
        arrays must have the same shapes.
        Args:
            y: has to be an numpy.array, a vector of shape m * 1.
            y_hat: has to be an numpy.array, a vector of shape m * 1.
            theta: has to be a numpy.array, a vector of shape n * 1.
            lambda_: has to be a float.
        Return:
            The element-wise regularized loss as a numpy array.
            None if y, y_hat, or theta are empty numpy.array.
            None if y and y_hat do not share the same shapes.
            None if y or y_hat is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            loss = (y - y_hat) ** 2
            reg = self.lambda_ * self.l2()
            return 0.5 * (loss + reg) / y.shape[0]
        except:
            # If something unexpected happened, we juste leave
            print("Something wrong during loss_elem.", file=sys.stderr)
            return None


    @check_types
    @check_measurement('x', 'y')
    def gradient_(self, x, y):
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
            x_ = np.hstack((np.ones((x.shape[0], 1)), x))
            theta_ = np.copy(self.thetas)
            theta_[0] = 0
            reg_grad = x_.T @ ((x_ @ self.thetas) - y) + self.lambda_ * theta_
            return reg_grad / y.shape[0]
        except:
            # If something unexpected happened, we juste leave
            print("Something wrong during gradient calculation.",
                  file=sys.stderr)
            return None

    @check_types
    @check_measurement('x', 'y')
    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.array, a vector of shape m * 1:
               (number of training examples, 1).
            y: has to be a numpy.array, a vector of shape m * 1:
               (number of training examples, 1).
            theta: has to be a numpy.array, a vector of shape 2 * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during
                      the gradient descent
        Return:
            self: instance of MyLinearRegression, more convenient in ex10.
            None if there is a matching shape problem.
            None if x, y, theta, alpha or max_iter is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # Performing the gradient descent
            for _ in range(self.max_iter):
                grad = self.gradient_(x, y)
                self.thetas = self.thetas - self.alpha * grad
            return self
        except:
            # If something unexpected happened, we juste leave
            print("Something wrong during fit.", file=sys.stderr)
            return None


# ########################################################################### #
#                                    Main                                     #
# ########################################################################### #
if __name__ == "__main__":
    pass