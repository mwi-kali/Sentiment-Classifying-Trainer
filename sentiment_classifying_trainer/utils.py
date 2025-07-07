import torch

import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    probs = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()
    intensity = probs.max(axis=1).mean() 
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "sentiment_intensity": intensity,
    }


def log_confusion_matrix(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix\n", cm)
