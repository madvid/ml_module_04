# Metrics related methods only.
import sys
import numpy as np

eps = 1e-8
PREC = 4
# ########################################################################## #
#  ______________________________ FUNCTIONS ________________________________ #
# ########################################################################## #


def _check_type_(y, ypred):
    """ Checking y and ypred are both numpy ndarray objects
    """
    # Checking type
    if (not isinstance(y, np.ndarray)) \
            or (not isinstance(ypred, np.ndarray)):
        s = "Unexpected type for y or ypred."
        print(s, file=sys.stderr)
        sys.exit()
    # Checking data type
    if y.dtype.kind != ypred.dtype.kind:
        s = "Unmatching data type."
        print(s, file=sys.stderr)
        sys.exit()


def _check_shape_(y, ypred):
    """ Checking y and ypred are both of the same shape
    and are both 1 or 2 dimensional + same dimension (in fact it is redondant).
    """
    # Checking shape along axis 0
    if y.shape != ypred.shape:
        s = "Mismatching shape between y and ypred."
        print(s, file=sys.stderr)
        sys.exit()
    # Checkin data type
    if (y.ndim > 2) \
            or (y.ndim > 2) \
            or (y.ndim != ypred.ndim):
        s = "Unconsistent dimension between y and ypred."
        # Well it should never happens, it would be catch by previous if.
        print(s, file=sys.stderr)
        sys.exit()


def _check_samples_(y, ypred):
    """ Checking the set of values of ypred respectively to y.
    (ie all the value in ypred must appear in y otherwise there is a problem)
    """
    set_y = np.unique(y)
    set_ypred = np.unique(ypred)
    if any([e not in set_y for e in set_ypred]):
        s = "Unexpected value in y_hat."
        print(s, file=sys.stderr)
        sys.exit()


def accuracy_score_(y: np.ndarray, ypred: np.ndarray, pos_label: int = 1):
    """
    Compute the accuracy score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
    Return:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    Reminder:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        with:
            TP: True Positive
            TN: True Negative
            FP: False Positive
            FN: True Negative
    """
    _check_type_(y, ypred)
    _check_shape_(y, ypred)
    _check_samples_(y, ypred)
    tp_arr = (y == pos_label) & (ypred == pos_label)
    fp_arr = (y != pos_label) & (ypred == pos_label)
    tn_arr = (y != pos_label) & (ypred != pos_label)
    fn_arr = (y == pos_label) & (ypred != pos_label)
    tp = tp_arr.sum()
    fp = fp_arr.sum()
    tn = tn_arr.sum()
    fn = fn_arr.sum()
    if (tp == 0) & (fp == 0) & (tn == 0) & (fn == 0):
        accuracy = 0
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return round(accuracy, PREC)


def precision_score_(y: np.ndarray, ypred: np.ndarray, pos_label: int = 1):
    """
    Compute the accuracy score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
    Return:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    Reminder:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        with:
            TP: True Positive
            TN: True Negative
            FP: False Positive
            FN: True Negative
    """
    _check_type_(y, ypred)
    _check_shape_(y, ypred)
    _check_samples_(y, ypred)
    tp_arr = (y == pos_label) & (ypred == pos_label)
    fp_arr = (y != pos_label) & (ypred == pos_label)
    tp = tp_arr.sum()
    fp = fp_arr.sum()
    precision = tp / (tp + fp + eps)
    return round(precision, PREC)


def recall_score_(y: np.ndarray, ypred: np.ndarray, pos_label: int = 1):
    """
    Compute the recall score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
    Return:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    Reminder:
        recall = TP / (TP + FN)
        with:
            TP: True Positive
            TN: True Negative
            FP: False Positive
            FN: False Negative
    """
    _check_type_(y, ypred)
    _check_shape_(y, ypred)
    _check_samples_(y, ypred)
    tp_arr = (y == pos_label) & (ypred == pos_label)
    fn_arr = (y == pos_label) & (ypred != pos_label)
    tp = tp_arr.sum()
    fn = fn_arr.sum()
    recall = tp / (tp + fn + eps)
    return round(recall, PREC)


def specificity_score_(y: np.ndarray, ypred: np.ndarray, pos_label: int = 1):
    """
    Compute the specificity score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
    Return:
        The specificity score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    Reminder:
        specificity = TN / (TN + FP)
        with:
            TP: True Positive
            TN: True Negative
            FP: False Positive
            FN: True Negative
    """
    _check_type_(y, ypred)
    _check_shape_(y, ypred)
    _check_samples_(y, ypred)
    tp_arr = (y == pos_label) & (ypred == pos_label)
    fp_arr = (y != pos_label) & (ypred == pos_label)
    tn_arr = (y != pos_label) & (ypred != pos_label)
    fn_arr = (y == pos_label) & (ypred != pos_label)
    fp = fp_arr.sum()
    tn = tn_arr.sum()
    specificity = tn / (tn + fp + eps)
    return round(specificity, PREC)


def f1_score_(y: np.ndarray, ypred: np.ndarray, pos_label: int = 1):
    """
    Compute the f1 score.
    Args:
        y:a [numpy.array] for the correct labels
        y_hat: [numpy.array] for the predicted labels
        pos_label: [str, int], class on which f1_score is reported (default=1)
    Return:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    _check_type_(y, ypred)
    _check_shape_(y, ypred)
    _check_samples_(y, ypred)
    precision = precision_score_(y, ypred, pos_label)
    recall = recall_score_(y, ypred, pos_label)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return round(f1, PREC)


def metrics_report(y: np.ndarray, ypred: np.ndarray):
    """ Displays the accuracy/recall/precision/f1 scores
    """
    accuracy = accuracy_score_(y, ypred)
    recall = recall_score_(y, ypred)
    precision = precision_score_(y, ypred)
    f1 = f1_score_(y, ypred)
    print("Accuracy:".ljust(15), accuracy)
    print("Recall:".ljust(15), recall)
    print("Precision:".ljust(15), precision)
    print("F1:".ljust(15), f1)
