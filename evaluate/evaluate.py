from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def compute_score(y_trues, y_preds):
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for y_true, y_pred in zip(y_trues, y_preds):

        accuracy.append(accuracy_score(y_true, y_pred))
        precision.append(precision_score(y_true, y_pred, average= 'macro', zero_division= 0))
        recall.append(recall_score(y_true, y_pred, average= 'macro', zero_division= 0))
        f1.append(f1_score(y_true, y_pred, average= 'macro', zero_division= 0))



    return np.array(accuracy).mean(), np.array(precision).mean(), np.array(recall).mean(), np.array(f1).mean()