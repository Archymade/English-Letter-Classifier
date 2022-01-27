''' Metrics for classification task. '''

from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score


def get_precision_score(y_train, y_hat, avg='macro', num_places=None, text=False):
    """ Get precision scores for model results. """

    score = precision_score(y_train, y_hat, average=avg)

    if num_places:
        score = round(score, num_places)
        msg = f"\t> Precision score (.{num_places}f) : \n\t\tAverage : {avg.title()}\n\t\tScore : [ {score} ]"
    else:
        l = len(str(score))
        msg = f"\t> Precision score (.{l}f) : \n\t\tAverage : {avg.title()}\n\t\tScore :  : [ {score} ]"

    return msg if text else score


def get_accuracy_score(y_train, y_hat, num_places=None, text=False):
    """ Get accuracy score for model results. """

    score = accuracy_score(y_train, y_hat)

    if num_places:
        score = round(score, num_places)
        msg = f"\t> Accuracy score (.{num_places}f) : \n\t\tScore : [ {score} ]"
    else:
        l = len(str(score))
        msg = f"\t> Accuracy score (.{l}f) : \n\t\tScore :  : [ {score} ]"

    return msg if text else score


def get_recall_score(y_train, y_hat, avg='macro', num_places=None, text=False):
    """ Get recall score for model results. """

    score = recall_score(y_train, y_hat, average=avg)

    if num_places:
        score = round(score, num_places)
        msg = f"\t> Recall score (.{num_places}f) : \n\t\tAverage : {avg.title()}\n\t\tScore : [ {score} ]"
    else:
        l = len(str(score))
        msg = f"\t> Recall score (.{l}f) : \n\t\tAverage : {avg.title()}\n\t\tScore :  : [ {score} ]"

    return msg if text else score


def get_f1_score(y_train, y_hat, avg='macro', num_places=None, text=False):
    """ Get f1 score for model results. """

    score = f1_score(y_train, y_hat, average=avg)

    if num_places:
        score = round(score, num_places)
        msg = f"\t> F1-score (.{num_places}f) : \n\t\tAverage : {avg.title()}\n\t\tScore : [ {score} ]"
    else:
        l = len(str(score))
        msg = f"\t> F1-score (.{l}f) : \n\t\tAverage : {avg.title()}\n\t\tScore :  : [ {score} ]"

    return msg if text else score


def performance_report(y, y_hat, split='train'):
    """ Obtain diagnostics on model performance. """

    print(f'Classification Report for {split.upper()} split.')
    print('='*50)
    print(classification_report(y, y_hat))

    return None
