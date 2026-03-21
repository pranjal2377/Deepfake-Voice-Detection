import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
import numpy as np

from src.utils.config import RESULTS_DIR

class MetricsVisualizer:
    def __init__(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix", filename="confusion_matrix.png"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        filepath = os.path.join(RESULTS_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        return filepath
        
    def plot_precision_recall_curve(self, y_true, y_scores, title="Precision-Recall Curve", filename="pr_curve.png"):
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, marker='.')
        plt.title(title)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        filepath = os.path.join(RESULTS_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        return filepath
        
    def plot_roc_curve(self, y_true, y_scores, title="ROC Curve", filename="roc_curve.png"):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title(title)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        filepath = os.path.join(RESULTS_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        return filepath

    def plot_model_comparison(self, accuracies, models=['Audio', 'NLP', 'Combined'], filename="model_comparison.png"):
        plt.figure(figsize=(8, 5))
        sns.barplot(x=models, y=accuracies)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
        filepath = os.path.join(RESULTS_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        return filepath
