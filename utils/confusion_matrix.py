import numpy as np
import pandas as pd
import sys
from sklearn.metrics import confusion_matrix

# ########################################################################## #
#  ______________________________ FUNCTIONS ________________________________ #
# ########################################################################## #


def check_type(y: np.ndarray, y_hat: np.ndarray):
    """ Checking the type of y and y_hat and the data_type of the target
    vector and predicted vector.
    Args:
        y    : [np.ndarray] target vector
        y_hat: [np.ndarray] predicted target vector.
    Return:
        None
    Remark:
        The function will print an error message and quit the program if
        there is any issue.
    """
    # Checking type
    if (not isinstance(y, np.ndarray)) \
            or (not isinstance(y_hat, np.ndarray)):
        s = "Unexpected type for y or y_hat."
        print(s, file=sys.stderr)
        sys.exit()
    # Checking data type
    if y.dtype.kind != y_hat.dtype.kind:
        s = "Unmatching data type."
        print(s, file=sys.stderr)
        sys.exit()


def check_shape(y: np.ndarray, y_hat: np.ndarray):
    """ Checking the shape of the target and prediction vectors.
    If the shape are different, a error message is printed on stderr
    and the program is stopped.
    Args:
        y    : [np.ndarray] target vector
        y_hat: [np.ndarray] predicted target vector.
    Return:
        None
    Remark:
        The function will print an error message and quit the program if
        there is any issue.
    """
    # Checking shape along axis 0
    if y.shape[0] != y_hat.shape[0]:
        s = "Unconsistent length between y and y_hat."
        print(s, file=sys.stderr)
        sys.exit()
    # Checkin data type
    if (y.ndim > 2) \
            or (y.ndim > 2) \
            or (y.ndim != y_hat.ndim):
        s = "Unconsistent dimension between y and y_hat."
        print(s, file=sys.stderr)
        sys.exit()


def check_labels(y: np.ndarray, labels: np.ndarray):
    """ Checking if each unique values of y_hat is not a new value and
    exists in the target vector y.
    If the shape are different, a error message is printed on stderr
    and the program is stopped.
    Args:
        y    : [np.ndarray] target vector
        labels: [np.ndarray] predicted target vector.
    Return:
        None
    Remark:
        The function will print an error message and quit the program if
        there is any issue.
    """
    if labels.ndim != 1:
        s = "'labels' should not be a nested list."
        print(s, file=sys.stderr)
        sys.exit()

    set_y = np.unique(y)
    set_labels = np.unique(labels)

    if len(set_labels) == 0:
        s = "'labels' should contains at least one label."
        print(s, file=sys.stderr)
        sys.exit()
    if any([e not in set_y for e in set_labels]):
        s = "At least one label specified must be in y."
        print(s, file=sys.stderr)
        sys.exit()


def confusion_matrix_(y: np.ndarray, yhat: np.ndarray,
                      labels=None, df_option: bool = False):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y: [numpy.array] the correct labels
        y_hat: [numpy.array] for the predicted labels
        labels: [list[string]]list of labels to index the matrix (optional).
            May be used to reorder / select a subset of labels. (default=None)
        df_option: [pandas.DataFrame], if True the function will return
            a pandas DataFrame instead of a numpy array. (default=False)
    Return:
        [numpy.array/pandas.Dataframe] confusion matrix according to
            df_option value.
        None if any error.
    Raises:
        This function should not raise any Exception.
    """
    check_type(y, yhat)
    check_shape(y, yhat)
    if (labels is not None) and (not isinstance(labels, list)):
        s = "Unexpected value for labels parameters."
        print(s, file=sys.stderr)
        sys.exit()
    if df_option not in [True, False]:
        s = "Unexpected value for df_option parameters."
        print(s, file=sys.stderr)
        sys.exit()

    if labels is None:
        labels = np.unique(np.concatenate((y, yhat))).astype(object)
    else:
        labels = np.array(labels)
        check_labels(y, labels)
    matrix = pd.DataFrame(data=np.zeros((labels.shape[0], labels.shape[0])),
                          index=labels,
                          columns=labels, dtype=np.int64)

    for index in labels:
        mask = y == index
        for col in labels:
            nb = np.sum(yhat[mask] == col)
            matrix[col][index] = nb

    if df_option:
        return matrix
    else:
        return matrix.values


if __name__ == '__main__':
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog',
                      'bird'])
    print("\n# Example 1:")
    # My implementation
    res = confusion_matrix_(y, y_hat)
    # sklearn implementation
    expected = confusion_matrix(y, y_hat)
    # Output:
    print("My confusion matrix:\n", res)
    print("sklearn confusion matrix:\n", expected)

    print("\n# Example 2:")
    y_hat = np.array(['dog', 'dog', 'norminet', 'norminet', 'norminet',
                      'bird'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
    # My implementation
    res = confusion_matrix_(y, y_hat)
    # sklearn implementation
    expected = confusion_matrix(y, y_hat)
    # Output:
    print("My confusion matrix:\n", res)
    print("sklearn confusion matrix:\n", expected)

    print("\n# Example 3:")
    # My implementation
    res = confusion_matrix_(y, y_hat, labels=['dog', 'norminet'])
    # sklearn implementation
    expected = confusion_matrix(y, y_hat, labels=['dog', 'norminet'])
    # Ouput
    print("My confusion matrix:\n", res)
    print("sklearn confusion matrix:\n", expected)

    print("\n# Example 4:")
    # My implementation
    res = confusion_matrix_(y, y_hat, df_option=True)
    # sklearn implementation
    skl = confusion_matrix(y, y_hat)
    # Output:
    print("My confusion matrix:\n", res)
    print("sklearn confusion matrix (no option to have df form):\n", skl)

    print("\n# Example 5:")
    # My implementation
    res = confusion_matrix_(y, y_hat, labels=['bird', 'dog'], df_option=True)
    # sklearn implementation
    skl = confusion_matrix(y, y_hat, labels=['bird', 'dog'])
    # Output:
    print("My confusion matrix:\n", res)
    print("sklearn confusion matrix (no option to have df form):\n", skl)
