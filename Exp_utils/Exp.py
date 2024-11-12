import matplotlib.pyplot as plt
import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler as undr_sampling
from imblearn.over_sampling import RandomOverSampler as ovr_sampling
from imblearn.over_sampling import ADASYN

from .utils.time_utils import get_TimeStamp_str

from sklearn.model_selection import train_test_split
from .plt_.classification_plts import *

from .imputation.impute import imputer_

from .utils.helper import *
from .Scoring.Classification_metrics import *
from .models.Classification_models import *

from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder

def run_main(x, y, exp_str):
    a = run_prediction(models, x, y,
                       metrics_=metrics_,
                       save_true_vs_pred_conf=False, random_state_=123,
                       exp_title=exp_str)
    sampling_Exp(x, y, models, metrics=metrics_, exp_title=exp_str)
    CI_curve(x, y, models[1:2], k_folds=10, exp_str=exp_str)


def sampling_Exp(x, y, model_lst, metrics, exp_title):
    final_res_df = pd.DataFrame()
    for sampling in [SMOTE(), undr_sampling(), ovr_sampling(), ADASYN()]:
        X_res, y_res = sampling.fit_resample(x, y)
        a = run_prediction(model_lst, X_res, y_res,
                           metrics_=metrics,
                           save_true_vs_pred_conf=False, random_state_=123,
                           exp_title=exp_title + ' - Sampling_' + str(sampling).replace('(', '').replace(')', ''),
                           save_exl_res=False)

        CI_curve(X_res, y_res, models[1:2], k_folds=10,
                 exp_str=exp_title + " - Sampling_" + str(sampling).replace('(', '').replace(')', ''))

        method_ = str(sampling).replace('()', '')
        print('method_ = ', method_)
        a['Sampling method'] = method_

        final_res_df = pd.concat([final_res_df, a, empty_row(a.columns.values)])

    cols_ = final_res_df.columns.values.tolist()
    cols_.remove('Sampling method')

    final_res_df[['Sampling method', *cols_]].to_excel('res/%s - Sampling.xlsx' % exp_title,
                                                       index=False)  # 3-67178 sampling

def CI_curve(X, Y, classifier, k_folds=10, exp_str=''):
    if isinstance(classifier, list): classifier = classifier[0]
    scores = []
    means = []
    fig = plt.figure(figsize=(15, 10))
    for k_ in range(2, k_folds + 1):
        exp = [i * 100 for i in cross_val_score(classifier, X, Y, cv=k_)]
        means.append(np.mean(exp))
        scores.append(exp)
        # print('scores = ', len(exp), exp, means[-1])

    x = np.array(range(2, k_folds + 1))
    y = means

    error = []
    for k, s in enumerate(scores, 2):
        error.append( (1.96 * np.std(s)) / np.sqrt(k))
        # print(k, s, np.std(s), np.sqrt(k))
    error = np.array(error)

    # Calculate the upper and lower bounds
    y_upper = y + error
    y_lower = y - error

    # Plotting the line
    plt.plot(x, y, '-k', label='Mean')

    # Plotting the area around the line
    plt.fill_between(x, y_lower, y_upper, alpha=0.5, label='Confidence Interval')

    plt.xlabel('Folds', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)

    plt.ylim((0, 100))

    plt.legend()

    file_name = 'res/CI %s %s - %s' % (str(classifier).replace('(', '').replace(')', ''), exp_str, get_TimeStamp_str())

    if exp_str != '':
        fig.savefig(file_name + '.pdf', bbox_inches='tight')
    plt.close()

    with open('results.txt', 'a') as f:
        f.write('\n\n' + file_name[4:] + ':  ')
        f.write(', '.join([str(l) for l in scores]))

def run_prediction(models_lst, x, y, metrics_,
                   save_true_vs_pred_conf=False,
                   exp_title=None, cols=None, random_state_=123, save_exl_res=True):

    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123, stratify=y)

    if exp_title is not None:
        exp_title = exp_title.replace('(', '').replace(')', '')

    exp_time = get_TimeStamp_str() + (' - ' + exp_title if exp_title is not None else '')
    print(exp_title)
    print("res/%s - Final_results - %s.xlsx" % (exp_title, exp_time))
    # return ''

    final_df = pd.DataFrame()
    true_vs_pred = pd.DataFrame()
    true_vs_pred['True'] = y_test
    m = None
    for co, m in enumerate(models_lst, 1):
        file_name = 'res/*** - %s - %s.csv' % (str(m).replace('<', '').replace('>', '').split('(')[0], exp_time)
        # if cols is not None and isinstance(m, catboost.CatBoostClassifier):
        #     m = None
        #     # print('cols = ', cols, 'intersected:', np.intersect1d(cat_feature, cols).tolist())
        #     m = catboost.CatBoostClassifier(verbose=0, cat_features=np.intersect1d(cat_feature, cols).tolist())
        # if hasattr(m, 'random_state'):
        #     m.random_state = random_state_
        # try:...
        #     # print(m, 'm.randomstate = ', m.random_state, 'df.shape:', X_train.shape)
        # except:
        #     ...
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        # print('accuracy_score:', accuracy_score(y_test, pred))
        y_train_pred = m.predict(X_train)

        predict_proba = None
        if hasattr(m, 'predict_proba'):
            predict_proba = m.predict_proba(X_train)

        res_train, df_train = get_results_binary_class(y_train, y_train_pred, pred_proba=predict_proba,
                                                       metric_lst=metrics_)

        df_train = pd.concat([df_train, df_train.describe().loc[['mean'], :]])
        df_train.columns = ['train_%s' % i for i in df_train.columns.values]

        predict_proba = None
        if hasattr(m, 'predict_proba'):
            predict_proba = m.predict_proba(X_test)
            # print('predict_proba.shape ----------', predict_proba.shape)

        res_test, df_test = get_results_binary_class(y_test, pred, pred_proba=predict_proba, metric_lst=metrics_)

        df_test = pd.concat([df_test, df_test.describe().loc[['mean'], :]])
        df_test = pd.concat([df_test, df_train], axis=1)
        df_test['model'] = str(m).split('(')[0]

        # conf matrix:
        cm_analysis(y_test, pred, labels=[0, 1], file_name=file_name[:-4].replace('***', 'CM') + '.pdf', figsize=(5, 5))

        # ROC curve
        ROC(m, X_test, y_test, file_name=file_name[:-4].replace("***", "ROC") + ".pdf")

        final_df = pd.concat([final_df, df_test])
        final_df = final_df.append(pd.Series(), ignore_index=True)
        # final_df.to_excel('res/%s/tmp - %s.xlsx' % (exp_time, exp_time), index=False)
        final_df['label'] = final_df['label'].apply(lambda x: 'Mean' if x == 0.5 else x)

    test_cols = ['model', 'label', 'recall_score', 'precision_score', 'f1_score', 'specificity', 'roc_auc_score',
                 'accuracy_score']
    final_df = final_df[[*test_cols, *df_train.columns.values]]
    if save_exl_res: final_df.to_excel('res/%s - Final_results - %s.xlsx' % (exp_title, exp_time), index=False)
    return final_df#[test_cols]


# def run_prediction_old(models_lst, X_train, X_test, y_train, y_test, metrics_, exp_title=None, cols=None):
#     exp_time = get_TimeStamp_str() + (' - ' + exp_title if exp_title is not None else '')

#     if os.path.isdir('results'):
#         os.mkdir('results/%s' % exp_time)
#     else:
#         os.mkdir('results')
#         os.mkdir('results/%s' % exp_time)
#     print('exp_time = ', exp_time)

#     final_df = pd.DataFrame()
#     true_vs_pred = pd.DataFrame()
#     true_vs_pred['True'] = y_test
#     m = None
#     for co, m in enumerate(models_lst, 1):
#         if cols is not None and isinstance(m, catboost.CatBoostClassifier):
#             m = None
#             print('cols = ', cols, 'intersected:', np.intersect1d(cat_feature, cols).tolist())
#             m = catboost.CatBoostClassifier(verbose=0, cat_features=np.intersect1d(cat_feature, cols).tolist())

#         m.fit(X_train, y_train)
#         pred = m.predict(X_test)
#         y_train_pred = m.predict(X_train)

#         res, df = get_results_binary_class(y_test, pred, metric_lst=metrics_)
#         df = pd.concat([df, df.describe().loc[['mean'], :]])
#         df['model'] = str(m).split('(')[0]

#         true_vs_pred['pred'] = pred
#         # print('results/%s/Conf_matrix - %s - %s%s.csv' % ( exp_time,   str(m).split('(')[0], exp_time, ' - ' + exp_title if exp_title is not None else ''))
#         true_vs_pred.to_csv('results/%s/Conf_matrix - %s - %s.csv' % (exp_time,
#                                                                       str(m).replace('<', '').replace('>', '').split(
#                                                                           '(')[0], exp_time), index=False)

#         final_df = pd.concat([final_df, df])
#         final_df = final_df.append(pd.Series(), ignore_index=True)
#         final_df.to_excel(
#             'results/%s/tmp - %s.xlsx' % (exp_time, exp_time),
#             index=False)
#         print('\r', '%d/%d %s' % (co, len(models_lst), str(m).split('(')[0]), end='')

#         final_df['label'] = final_df['label'].apply(lambda x: 'Mean' if x == 0.5 else x)

#     final_df = final_df[
#         ['model', 'label', 'recall_score', 'precision_score', 'f1_score', 'specificity', 'roc_auc_score',
#          'accuracy_score']]
#     final_df.to_excel(
#         'results/%s/Final_results - %s.xlsx' % (exp_time, exp_time),
#         index=False)
#     return final_df, m
# def CI_curve(X, Y, classifier, k_folds=10, exp_str=''):
#     if isinstance(classifier, list): classifier = classifier[0]
#     scores = []
#     for k_ in range(2,k_folds+1):
#         scores.append ([i * 100 for i in cross_val_score(classifier, X, Y, cv=k_)])

    # fig = plt.figure(figsize=(15, 10))
    # # Create the data set
    # x = np.arange(1, k_folds + 1, 1)
    # y = scores  # this the accuracies from the cros_val_score method
    #
    # # Define the confidence interval
    # ci = 10 * np.std(y) / np.mean(y)
    # # print( 'ci = ' , ci )
    #
    # # Plot the accuracies vs the k-fold
    # plt.plot(x, y)
    # # print( 'y = ' , y )
    #
    # # Set the font size of xticks
    # plt.xticks(fontsize=18)
    #
    # # Set the font size of xticks
    # plt.yticks(fontsize=18)
    #
    # # Plot the confidence interval
    # plt.fill_between(x, (y - ci), (y + ci), color='blue', alpha=0.2)
    # plt.ylabel('Model Accuracy', fontname="Brush Script MT", fontsize=18)
    # plt.xlabel('K iterations', fontname="Brush Script MT", fontsize=18)
    #
    # file_name = 'res/CI %s %s - %s' % (str(classifier).replace('(', '').replace(')', ''), exp_str, get_TimeStamp_str())
    # if exp_str != '':
    #     fig.savefig(file_name + '.pdf', bbox_inches='tight')
    # plt.close()
    #
    # with open('%s.txt' % file_name, 'w') as f:
    #     f.write('\n'.join([str(l) for l in scores]))
    # return scores


    # -------------------------------------------------------------------------------
    # --------------------  Ahmed Salah Code  ---------------------------------------
    # -------------------------------------------------------------------------------
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # f2 = [224.9557507, 248.0768616]
    # f3 = [284.9969039, 280.0744314, 282.9383881]
    # f4 = [268.1327537, 163.1605529, 326.6287923, 150.5982218]
    # f5 = [288.6565169, 198.1170005, 31.6877166, 377.8948111, 96.1501187]
    #
    # x = np.array([2, 3, 4, 5])
    # y = np.array([np.mean(f2), np.mean(f3), np.mean(f4), np.mean(f5)])
    # error = np.array([1.96 * np.std(f2) / np.sqrt(2), 1.96 * np.std(f3) / np.sqrt(3), 1.96 * np.std(f4) / np.sqrt(4),
    #                   1.96 * np.std(f5) / np.sqrt(5)])
    #
    # # Calculate the upper and lower bounds
    # y_upper = y + error  # 3 fold [140, 130 , 120] y = cv_mean, cv_std then calc  error =  confidence_interval = 1.96 * cv_std / np.sqrt(10)
    # y_lower = y - error
    #
    # # Plotting the line
    # plt.plot(x, y, '-k', label='Mean')
    #
    # # Plotting the area around the line
    # plt.fill_between(x, y_lower, y_upper, color='gray', alpha=0.5, label='Confidence Interval')
    #
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('Line with Confidence Interval')
    # plt.legend()
    # plt.show()


# from ..Scoring.Classification_metrics import metrics_
