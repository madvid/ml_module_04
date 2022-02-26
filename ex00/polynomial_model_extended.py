import sys
import numpy as np
from numpy.polynomial.polynomial import polyvander

np.set_printoptions(suppress=True)  # supppress the scientific formating


# ########################################################################## #
#                             Function definitions                           #
# ########################################################################## #
def add_polynomial_features(x, power):
    """Add polynomial features to matrix x by raising its columns to every
    power in the range of 1 up to the power given in argument.
    Args:
        x: has to be an numpy.array, where x.shape = (m,n) i.e. a matrix of
            shape m * n.
        power: has to be a positive integer, the power up to which the columns
            of matrix x are going to be raised.
    Return:
        - The matrix of polynomial features as a numpy.array, of shape
            m * (np), containg the polynomial feature values for all training
            examples.
        - None if x is an empty numpy.array.
        - None if x or power is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        # Checking type
        if not isinstance(x, np.ndarray):
            s = "Unexpected type for x."
            print(s, file=sys.stderr)
            return None
        # Checking data type, 'i': signed integer, 'u': unsigned integer,
        # 'f': float
        if x.dtype.kind not in ["i", "u", "f"]:
            s = "Unexpected data type for x."
            print(s, file=sys.stderr)
            return None
        # Checking type of power
        if (not isinstance(power, int)) and (power < 0):
            s = "Unexpected type or value for power."
            print(s, file=sys.stderr)
            return None

        x_ = x
        lst_vander = []
        if x.ndim == 1:
            x_ = np.expand_dims(x_, axis=1)
        for ii in range(x_.shape[1]):
            lst_vander.append(polyvander(x_[:, ii], power))
        res = x_
        for ii in range(x.shape[1], power + x.shape[1]):
            for jj in range(len(lst_vander)):
                res = np.hstack((res, lst_vander[jj][:, ii : ii + 1]))
        return res.astype(x.dtype)
    except:
        return None


# ########################################################################## #
#                                      Main                                  #
# ########################################################################## #
if __name__ == "__main__":
    x = np.arange(1, 11).reshape(5, 2)

    print("# Example 1:")
    my_res = add_polynomial_features(x, 3)
    expected_res = np.array(
        [
            [1, 2, 1, 4, 1, 8],
            [3, 4, 9, 16, 27, 64],
            [5, 6, 25, 36, 125, 216],
            [7, 8, 49, 64, 343, 512],
            [9, 10, 81, 100, 729, 1000],
        ]
    )
    print("my_res:\n", my_res)
    print("expected_res:\n", expected_res)

    print("\n# Example 2:")
    my_res = add_polynomial_features(x, 5)
    expected_res = np.array(
        [
            [1, 2, 1, 4, 1, 8, 1, 16],
            [3, 4, 9, 16, 27, 64, 81, 256],
            [5, 6, 25, 36, 125, 216, 625, 1296],
            [7, 8, 49, 64, 343, 512, 2401, 4096],
            [9, 10, 81, 100, 729, 1000, 6561, 10000],
        ]
    )
    print("my_res:\n", my_res)
    print("expected_res:\n", expected_res)
