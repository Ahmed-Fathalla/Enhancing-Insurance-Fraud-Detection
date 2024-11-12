import matplotlib.pyplot as plt
import traceback
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

def ROC(m, X_test, y_test, file_name=None):
    try:
        pos_lbl = 1
        RocCurveDisplay.from_estimator(m, X_test, y_test, pos_label=pos_lbl)
        # print('m.predict_proba(X_test) = ', m.predict_proba(X_test).shape)
        # print('roc_auc_score = ', roc_auc_score(y_test, m.predict_proba(X_test)[:,pos_lbl].reshape(-1,1)))
        if file_name is not None:
            plt.savefig(file_name , bbox_inches='tight')
        plt.close()
    except:
        print('\n**** Err:\n', traceback.format_exc())

def cm_analysis(y_true, y_pred, labels=None, ymap=None, file_name=None, figsize=(5,5)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]

    labels = list(set(y_true)) if labels is None else labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/\n%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'True label'
    cm.columns.name = 'Predicted label'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cmap="Blues", annot_kws={"size": 15} )
    if file_name is not None:
        plt.savefig(file_name , bbox_inches='tight')
    plt.close()