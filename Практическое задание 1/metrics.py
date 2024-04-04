import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def AUC(y_true, y_score):
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        True labels.

    y_score: array-like of shape (n_samples,)
        Target scores.

    Returns
    -------
    AUC: float
        Area Under the Curve score.
    """

    return roc_auc_score(y_true, y_score)


def Accuracy(y_true, y_pred):
    """
    Compute Accuracy.

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        True labels.

    y_pred: array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    Accuracy: float
        Accuracy OMG.
    """

    return accuracy_score(y_true, y_pred)


def NUM(y_true, y_pred):
    """
    Compute the number of incorrectly classified samples (int).

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        True labels.

    y_pred: array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    NUM: int
        Number of incorrectly classified samples.
    """

    return len(y_true) - accuracy_score(y_true, y_pred, normalize=False)


def ASY(y_true, y_pred, P):
    """
    Compute asymmetric penalty for incorrect classification.
    For penalty matrix P penalty is sum of p_{y_true_i, y_pred_i} for i in n_samples .

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        True labels.

    y_pred: array-like of shape (n_samples,)
        Predicted labels.

    P: array-like of shape (n_classes, n_classes)
        Penalty matrix, i.e. p_ij is a penalty for classifying an object of class i to class j.

    Returns
    -------
    ASY: float
        Asymmetric penalty for incorrect classification.
    """

    return P[y_true.astype(int), y_pred.astype(int)].sum()


def ASY1(y_true, y_pred):
    """
    Compute asymmetric penalty for incorrect classification with penalty matrix P = [[-9, 9], [1, 0]].

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        True labels.

    y_pred: array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    ASY1: float
        Asymmetric penalty for incorrect classification.
    """

    return ASY(y_true, y_pred, np.array([[-9, 9], [1, 0]]))


def RASY1(y_true, y_pred):
    """
    Compute RELATIVE asymmetric penalty for incorrect classification with penalty matrix P = [[-9, 9], [1, 0]].

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        True labels.

    y_pred: array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    RASY1: float
        RELATIVE asymmetric penalty for incorrect classification.
    """

    return ASY1(y_true, y_pred) / len(y_true)


def ASY2(y_true, y_pred):
    """
    Compute asymmetric penalty for incorrect classification with penalty matrix P = [[-1, 3], [2, -1]].

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        True labels.

    y_pred: array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    ASY2: float
        Asymmetric penalty for incorrect classification.
    """

    return ASY(y_true, y_pred, np.array([[-1, 3], [2, -1]]))


def RASY2(y_true, y_pred):
    """
    Compute RELATIVE asymmetric penalty for incorrect classification with penalty matrix P = [[-1, 3], [2, -1]].

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        True labels.

    y_pred: array-like of shape (n_samples,)
        Predicted labels.

    Returns
    -------
    RASY2: float
        RELATIVE asymmetric penalty for incorrect classification.
    """

    return ASY2(y_true, y_pred) / len(y_true)
