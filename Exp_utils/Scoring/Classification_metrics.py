import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import recall_score, \
                            precision_score, \
                            f1_score, \
                            accuracy_score,  \
                            roc_auc_score, \
                            classification_report,  \
                            confusion_matrix
from ..utils.utils import df_to_file


def specificity(y_true: np.array, y_pred: np.array, labels: set = None, pos_label=None):
    # https://stackoverflow.com/questions/33275461/specificity-in-scikit-learn
    #   Remembering that in binary classification,
    #       recall of the positive class is also known as “sensitivity”;
    #       recall of the negative class is “specificity”
    if labels is None: # Determine classes from the values
        labels = set(np.concatenate((np.unique(y_true), np.unique(y_pred))))
    label = labels.copy()
    label.remove(pos_label)
    return recall_score(y_true, y_pred, labels=labels , pos_label=label[0])


metrics_ = [               recall_score,   precision_score,   f1_score,   specificity,   roc_auc_score,   accuracy_score]
metrics_label = ['label', 'recall_score', 'precision_score', 'f1_score', 'specificity', 'roc_auc_score', 'accuracy_score']

def get_results_binary_class(y_true, y_pred, pred_proba= None, metric_lst=metrics_):
    labels=list(set(y_true))
    res_1, res_2 = [labels[0]], [labels[1]]

    for m in metric_lst[:4]: #[:3]
        res_1.append(m(y_true, y_pred,labels=labels , pos_label=labels[0]))
        res_2.append(m(y_true, y_pred,labels=labels , pos_label=labels[1]))

    if pred_proba is not None:
        roc_0 = roc_auc_score(y_true, pred_proba[:,0])
        roc_1 = roc_auc_score(y_true, pred_proba[:,1])
    else:
        roc_0 = roc_1 = -1
    res_1 =  res_1 + [roc_0, accuracy_score(y_true, y_pred)]
    res_2 =  res_2 + [roc_1, accuracy_score(y_true, y_pred)]
    out = []

    out.append(res_1)
    out.append(res_2)
    df = df_to_file(out, cols=metrics_label, fold_col=None, round_=5, save_to_file=None, padding='left',
                    rep_newlines='\t', print_=False, wide_col='', pre='', post='')
    return out, df
