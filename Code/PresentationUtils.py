from typing import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score


def plot_pre_rec_curve(y_test: pd.Series, probabilities: pd.Series) -> NoReturn:
    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the precision-recall curve for the model
    plt.plot(recall, precision, marker='.')
    average_precision = average_precision_score(y_test, probabilities)
    plt.title('Precision-Recall curve: \nAverage Precision-Recall Score = {0:0.2f}'.format(
        average_precision), fontsize=16)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the plot
    plt.show()


def plot_roc(y_test: pd.Series, probabilities: pd.Series) -> NoReturn:
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    # plot 50/50 line
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.title('ROC curve:', fontsize=16)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the plot
    plt.show()


def format_classification_report(y_test: pd.Series, predictions: pd.Series) -> pd.DataFrame:
    labels = ['fake', 'real']
    return pd.DataFrame(classification_report(y_test, predictions, target_names=labels, output_dict=True)).transpose()


def format_confusion_matrix(y_test: pd.Series, predictions: pd.Series) -> pd.DataFrame:
    conf_matrix = pd.DataFrame(confusion_matrix(y_test, predictions, labels=[0, 1]),
                               index=['True_Fake', 'True_Real'], columns=['Predict_Fake', 'Predict_Real'])

    conf_matrix['True_Totals'] = [sum(row) for row in conf_matrix.values]
    return conf_matrix
