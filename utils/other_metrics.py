import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Classif_utils():
    @staticmethod
    def _true_positive_(y, y_hat):
        pass
    
    @staticmethod
    def _true_negative_(y, y_hat):
        pass

    @staticmethod
    def _false_positive_(y, y_hat):
        pass

    @staticmethod
    def _false_negative_(y, y_hat):
        pass


def check_type(y, y_hat):
    """
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


def check_shape(y, y_hat):
    """
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


def check_samples(y, y_hat):
    """
    """
    set_y = np.unique(y)
    set_y_hat = np.unique(y_hat)
    
    if any([e not in set_y for e in set_y_hat]):
        s = "Unexpected value in y_hat."
        print(s, file=sys.stderr)
        sys.exit()


def labelencode(y, y_hat):
    """
    """
    u, y_encoded = np.unique(y, return_inverse=True)
    masks = [np.where(y_hat == label) for label in u]
    y_hat_encoded = np.zeros(y_encoded.shape)
    for ii, mask in enumerate(masks):
        y_hat_encoded[mask] = ii

    return y_encoded, y_hat_encoded


def accuracy_score_(y, y_hat):
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
        accuracy = (TP + TN) / (TP + TF + FP + FN)
        with:
            TP: True Positive
            TN: True Negative
            FP: False Positive
            FN: True Negative
    """
    try:
        check_type(y, y_hat)
        check_shape(y, y_hat)
        check_samples(y, y_hat)
        if y.dtype.kind == 'U': # testing if kind is unicode (string)
            y_, y_hat_ = labelencode(y, y_hat)
        else:
            y_, y_hat_ = y, y_hat
        tp = Classif_utils._true_positive_(y_, y_hat_)
        tn = Classif_utils._true_negative_(y_, y_hat_)
        fp = Classif_utils._false_positive_(y_, y_hat_)
        fn = Classif_utils._false_negative_(y_, y_hat_)
        accuracy = (tp + tn) / (fp + fn)
        return accuracy
    except:
        return None


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        check_type(y, y_hat)
        check_shape(y, y_hat)
        check_samples(y, y_hat)
        tp = Classif_utils._true_positive_(y, y_hat)
        fp = Classif_utils._false_positive_(y, y_hat)
        precision = tp / (tp + fp)
        return precision
    except:
        return None


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        check_type(y, y_hat)
        check_shape(y, y_hat)
        check_samples(y, y_hat)
        tp = Classif_utils._true_positive_(y, y_hat)
        fn = Classif_utils._false_negative_(y, y_hat)
        recall = tp / (tp + fn)
        return recall
    except:
        return None


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    # try:
    check_type(y, y_hat)
    check_shape(y, y_hat)
    check_samples(y, y_hat)
    tp = Classif_utils._true_positive_(y, y_hat)
    fp = Classif_utils._false_positive_(y, y_hat)
    fn = Classif_utils._false_negative_(y, y_hat)
    print(f"{tp} - {fp} - {fn}")
    f1 = 2 * tp / (2 * tp + fn + fp)
    return f1
    #except:
    #    return None


if __name__ == "__main__":
    # Example 1:
    y_hat = np.array([[1],[ 1],[ 0],[ 1],[ 0],[ 0],[ 1],[ 1]])
    y = np.array([[1],[ 0],[ 0],[ 1],[ 0],[ 1],[ 0],[ 0]])
    # Accuracy
    ## your implementation
    accuracy_score_(y, y_hat)
    ## Output:
    0.5
    ## sklearn implementation
    accuracy_score(y, y_hat)
    ## Output:
    0.5
    # Precision
    ## your implementation
    precision_score_(y, y_hat)
    ## Output:
    0.4
    ## sklearn implementation
    precision_score(y, y_hat)
    ## Output:
    0.4
    # Recall
    ## your implementation
    recall_score_(y, y_hat)
    ## Output:
    0.6666666666666666
    ## sklearn implementation
    recall_score(y, y_hat)
    ## Output:
    0.6666666666666666
    # F1-score
    ## your implementation
    f1_score_(y, y_hat)
    ## Output:
    0.5
    ## sklearn implementation
    f1_score(y, y_hat)
    ## Output:
    0.5

    # Example 2:
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
    # Accuracy
    ## your implementation
    accuracy_score_(y, y_hat)
    ## Output:
    0.625
    ## sklearn implementation
    accuracy_score(y, y_hat)
    ## Output:
    0.625
    # Precision
    ## your implementation
    precision_score_(y, y_hat, pos_label='dog')
    ## Output:
    0.6
    ## sklearn implementation
    precision_score(y, y_hat, pos_label='dog')
    ## Output:
    0.6
    # Recall
    ## your implementation
    recall_score_(y, y_hat, pos_label='dog')
    ## Output:
    0.75
    ## sklearn implementation
    recall_score(y, y_hat, pos_label='dog')
    ## Output:
    0.75
    # F1-score
    ## your implementation
    f1_score_(y, y_hat, pos_label='dog')
    ## Output:
    0.6666666666666665
    ## sklearn implementation
    f1_score(y, y_hat, pos_label='dog')
    ## Output:
    0.6666666666666665