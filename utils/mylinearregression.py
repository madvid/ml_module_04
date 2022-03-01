from __future__ import annotations
import numpy as np
from math import sqrt
import sys


class Metrics():

    @staticmethod
    def mse_(y, y_hat):
        """ Calculate the MSE between the predicted output and the
        real output.
        Args:
            y: has to be a numpy.array, a vector of shape m * 1.
            y_hat: has to be a numpy.array, a vector of shape m * 1.
        Returns:
            mse: has to be a float.
            None if there is a matching shape problem.
        Raises:
            This function should not raise any Exception.
        """
        try:
            mse = (1.0 / y.shape[0]) * np.sum((y - y_hat) ** 2, axis=0)
            return float(mse)
        except:
            return None

    @staticmethod
    def rmse_(y, y_hat):
        """ Calculate the RMSE between the predicted output and
        the real output.
        Args:
            y: has to be a numpy.array, a vector of shape m * 1.
            y_hat: has to be a numpy.array, a vector of shape m * 1.
        Returns:
            rmse: has to be a float.
            None if there is a matching shape problem.
        Raises:
            This function should not raise any Exception.
        """
        try:
            rmse = sqrt(Metrics.mse_(y, y_hat))
            return float(rmse)
        except:
            return None

    @staticmethod
    def mae_(y, y_hat):
        """ Calculate the MAE between the predicted output and
        the real output.
        Args:
            y: has to be a numpy.array, a vector of shape m * 1.
            y_hat: has to be a numpy.array, a vector of shape m * 1.
        Returns:
            mae: has to be a float.
            None if there is a matching shape problem.
        Raises:
            This function should not raise any Exception.
        """
        try:
            mae = (1.0 / y.shape[0]) * np.sum(np.absolute(y - y_hat), axis=0)
            return float(mae)
        except:
            return None

    @staticmethod
    def r2score_(y, y_hat):
        """ Calculate the R2score between the predicted output
        and the output.
        Args:
            y: has to be a numpy.array, a vector of shape m * 1.
            y_hat: has to be a numpy.array, a vector of shape m * 1.
        Returns:
            r2score: has to be a float.
            None if there is a matching shape problem.
        Raises:
            This function should not raise any Exception.
        """
        try:
            mean = np.mean(y, axis=0)
            residual = np.sum((y_hat - y) ** 2, axis=0)
            m_var = np.sum((y - mean) ** 2, axis=0)
            r2 = 1 - (residual / m_var)
            return float(r2)
        except:
            return None


class MyLinearRegression(Metrics):
    """ Homemade linear regression class to fit like a tiny boss-ish
    """
    CLS_loss_fct = Metrics.mse_

    def __init__(self, thetas, alpha=1e-2, max_iter=1000):
        # Checking of the attributes:
        if (not isinstance(thetas, (np.ndarray, tuple, list))) \
            or (not isinstance(alpha, (int, float))) \
                or (not isinstance(max_iter, int)):
            s = "At least one of the parameters is not of expected type."
            raise TypeError(s)

        # Conversion of thetas and testing the shape of the parameters.
        thetas = self._convert_thetas_(thetas)
        if (thetas.ndim != 2):
            s = "Unexpected number of dimension for thetas. Must be 2."
            print(s, file=sys.stderr)
            sys,exit()
        if (thetas.shape[1] != 1):
            s = "Unexpected shape for thetas. It must be n * 1 shape."
            print(s, file=sys.stderr)
            sys,exit()
        # Checking data type, 'i': signed integer, 'u': unsigned integer,
        # 'f': float
        if thetas.dtype.kind not in ["i", "u", "f"]:
            s = "Unexpected data type for theta."
            print(s, file=sys.stderr)
            return None
        # Checking the value of the learning rate
        if (alpha >= 1) or (alpha <= 0) or (max_iter <= 0):
            return None
        # Casting self.theta to float, in case it is integer
        self.thetas = thetas.astype('float64')
        self.alpha = float(alpha)
        self.max_iter = max_iter
        self.thetas = thetas

    @staticmethod
    def _convert_thetas_(thetas):
        """ Private function, convert thetas parameter in the constructor
        from tuple or list into numpy ndarray.
        Args:
            thetas: list, tuple or numpy ndarray containing the model's
                    coefficients.
        """
        if isinstance(thetas, np.ndarray):
            return thetas
        return np.array(thetas).reshape(-1, 1)

    def _gradient_(self, x, y):
        """ Private function gradient, there is no test performed on the
        parameters. It is to avoid to perform useless same tests at every
        iteration of the loop in the fit method.
        Args:
            x: has to be an numpy.array, a matrix of shape m * n.
            y: has to be an numpy.array, a vector of shape m * 1.
        Return:
            The gradient as a numpy.array, a vector of shape n * 1,
        """
        xp = np.hstack((np.ones((x.shape[0], 1)), x))
        return xp.T @ (xp @ self.thetas - y) / x.shape[0]

    def gradient(self, x, y):
        """Computes a gradient vector from three non-empty numpy.array,
        without any for-loop. The three arrays must have the compatible shapes.
        Args:
            x: has to be an numpy.array, a matrix of shape m * n.
            y: has to be an numpy.array, a vector of shape m * 1.
        Return:
            The gradient as a numpy.array, a vector of shape n * 1,
              containg the result of the formula for all j.
            None if x, y, or self.thetas are empty numpy.array.
            None if x, y and self.thetas do not have compatible shapes.
            None if x, y or self.thetas is not of expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # Testing the type of the parameters, numpy array expected.
            if (not isinstance(x, np.ndarray)) \
                or (not isinstance(y, np.ndarray)) \
                    or (not isinstance(self.thetas, np.ndarray)):
                return None

            # Testing the shape of the paramters.
            if (x.shape[1] + 1 != self.thetas.shape) \
                or (y.shape[1] != 1) \
                    or (self.thetas.shape[1] != 1) \
                    or (x.shape[0] != y.shape[0]):
                return None
            grad = self._gradient_(x, y)

            return grad
        except:
            # If something unexpected happened, we juste leave
            return None

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
            # Checking x, y and theta are numpy array
            if (not isinstance(x, np.ndarray)) \
                or (not isinstance(y, np.ndarray)) \
                    or (not isinstance(self.thetas, np.ndarray)):
                return None
            # Checking the shape of x, y and self.theta
            if (y.shape[1] != 1) \
                or (x.shape[0] != y.shape[0]) \
                    or (self.thetas.shape[0] != x.shape[1] + 1):
                return None
            # Performing the gradient descent
            for _ in range(self.max_iter):
                grad = self._gradient_(x, y)
                self.thetas = self.thetas - self.alpha * grad
            return self
        except:
            # If something unexpected happened, we juste leave
            print("Something wrong during fit.", file=sys.stderr)
            return None

    def loss_elem_(self, x, y):
        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            J_elem: numpy.array, a vector of dimension
                    (number of the training examples,1).
            None if there is a dimension matching problem between y and y_hat.
            None if y or y_hat is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # Checking y and y_hat are numpy array
            if (not isinstance(x, np.ndarray)) \
                    or (not isinstance(y, np.ndarray)):
                return None

            # Checking the shape of y and y_hat
            if (x.shape[1] + 1 != self.thetas.shape[0]) \
                    or (y.shape[1] != 1) \
                    or (x.shape[0] != y.shape[0]):
                return None

            res = (self.predict_(x) - y) ** 2
            return res
        except:
            None

    def loss_(self, x, y):
        """Computes the half mean squared error of two non-empty numpy.array,
        without any for loop. The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            The half mean squared error of the two vectors as a float.
            None if y or y_hat are empty numpy.array.
            None if y and y_hat does not share the same dimensions.
            None if y or y_hat is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            # Checking y and y_hat are numpy array
            if (not isinstance(x, np.ndarray)) \
                    or (not isinstance(y, np.ndarray)):
                return None

            # Checking the shape of y and y_hat
            if (x.shape[1] + 1 != self.thetas.shape[0]) \
                    or (y.shape[1] != 1) \
                    or (x.shape[0] != y.shape[0]):
                return None

            # loss = MyLinearRegression.CLS_loss_fct(y, y_hat)
            loss = (self.predict_(x) - y).T @ (self.predict_(x) - y)
            return float(loss) / (2.0 * y.shape[0])
        except:
            None

    @staticmethod
    def _loss_elem_(y, y_hat):
        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_elem: numpy.array, a vector of dimension
                (number of the training examples,1).
        None if there is a dimension matching problem between y and y_hat.
        None if y or y_hat is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        try:
            # Checking y and y_hat are numpy array
            if (not isinstance(y, np.ndarray)) \
                    or (not isinstance(y_hat, np.ndarray)):
                return None

            # Checking the shape of y and y_hat
            if (y.shape[1] != 1) \
                or (y_hat.shape[1] != 1) \
                    or (y_hat.shape[0] != y.shape[0]):
                return None

            res = (y - y_hat) ** 2
            return res
        except:
            None

    @staticmethod
    def _loss_(y, y_hat):
        """Computes the half mean squared error of two non-empty numpy.array,
        without any for loop. The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Returns:
            The half mean squared error of the two vectors as a float.
            None if y or y_hat are empty numpy.array.
            None if y and y_hat does not share the same dimensions.
            None if y or y_hat is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            loss = MyLinearRegression.CLS_loss_fct(y, y_hat)
            # loss = (y - y_hat).T @ (y - y_hat) / (2.0 * y.shape[0])
            return float(loss) / 2.0
        except:
            # If something unexpected happened, we juste leave
            None

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty
        numpy.array.
        Args:
            x: has to be an numpy.array, a vector of shape m * 1.
            theta: has to be an numpy.array, a vector of shape 2 * 1.
        Returns:
            y_hat as a numpy.array, a vector of shape m * 1.
            None if x or theta are empty numpy.array.
            None if x or theta shapes are not appropriate.
            None if x or theta is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            if not isinstance(x, (np.ndarray)):
                return None
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if any([n == 0 for n in x.shape]):
                return None
            if self.thetas.shape != (x.shape[1] + 1, 1):
                return None
            xp = np.hstack((np.ones((x.shape[0], 1)), x))
            ypred = xp @ self.thetas
            return ypred
        except:
            # If something unexpected happened, we juste leave
            return None


if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLinearRegression([[1.], [1.], [1.], [1.], [1]])

    print("# Example 0:")
    pred = mylr.predict_(X)
    # Output:
    expected_pred = np.array([[8.], [48.], [323.]])
    print("my prediction:".ljust(20), pred.reshape(1, -1))
    print("expected prediction:".ljust(20), pred.reshape(1, -1))

    print("\n# Example 1:")
    loss_e = mylr.loss_elem_(X, Y)
    # Output:
    expected_loss_e = np.array([[225.], [0.], [11025.]])
    print("my loss elem:".ljust(20), loss_e.reshape(1, -1))
    print("expected loss elem:".ljust(20), expected_loss_e.reshape(1, -1))

    print("\n# Example 2:")
    loss = mylr.loss_(X, Y)
    # Output:
    expected_loss = 1875.0
    print("my loss:".ljust(15), loss)
    print("expected loss:".ljust(15), expected_loss)

    print("\n# Example 3:")
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    # Output:
    expected_thetas = np.array([[18.188], [2.767], [-0.374], [1.392], [0.017]])
    print("my theta after training:\n", mylr.thetas.reshape(1, -1))
    print("expected theta after training:\n", expected_thetas.reshape(1, -1))

    print("\n# Example 4:")
    pred = mylr.predict_(X)
    # Output:
    expected_pred = np.array([[23.417], [47.489], [218.065]])
    print("my prediction:\n", pred.reshape(1, -1))
    print("expected prediction:\n", expected_pred.reshape(1, -1))

    print("\n# Example 5:")
    loss_e = mylr.loss_elem_(X, Y)
    # Output:
    expected_loss_e = np.array([[0.174], [0.260], [0.004]])
    print("my loss elem:\n", loss_e.reshape(1, -1))
    print("expected loss elem:\n", expected_loss_e.reshape(1, -1))

    print("\n# Example 6:")
    loss = mylr.loss_(X, Y)
    # Output:
    expected_loss = 0.0732
    print("my loss:".ljust(15), loss)
    print("expected loss:".ljust(15), expected_loss)
