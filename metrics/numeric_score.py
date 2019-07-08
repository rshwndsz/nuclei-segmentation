import numpy as np


def numeric_score(prediction, ground_truth):
    """
    Computes numeric scores

    FP = False Positives
    FN = False Negatives
    TP = True Positives
    TN = True Negatives

    :return: (FP, FN, TP, TN)
    """
    fp = np.float(np.sum((prediction == 1) & (ground_truth == 0)))
    fn = np.float(np.sum((prediction == 0) & (ground_truth == 1)))
    tp = np.float(np.sum((prediction == 1) & (ground_truth == 1)))
    tn = np.float(np.sum((prediction == 0) & (ground_truth == 0)))

    return fp, fn, tp, tn


def accuracy_score(prediction, ground_truth):
    """
    Computes model accuracy

    :param prediction: Predicted Tensor
    :param ground_truth: Ground truth Tensor
    :return: Accuracy in percentage
    """
    fp, fn, tp, tn = numeric_score(prediction, ground_truth)
    n = fp + fn + tp + tn
    accuracy = np.divide(tp + tn, n)
    return accuracy * 100.0
