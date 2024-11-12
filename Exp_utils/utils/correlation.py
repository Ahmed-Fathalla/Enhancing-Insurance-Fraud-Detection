import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn import preprocessing

def get_VIF(df, show_feat_name=True, return_df=False, report_file=None):

    object_feature = [i for i, j in zip(df.dtypes.index, df.dtypes) if j == 'object']
    if len(object_feature) > 0:
        for col in object_feature:
            label_encoder = preprocessing.LabelEncoder()
            df.loc[:, col] = label_encoder.fit_transform(df[col]).values

    vif_series = pd.Series([vif(df.values, i) for i in range(df.shape[1])], index=df.columns).reset_index()
    vif_series.columns = ['Feature_name', 'VIF']
    vif_series.sort_values('VIF',ascending=0)
    print(vif)

    if show_feat_name:
        print(vif_series.set_index('Feature_name'))
    else:
        vif_series.index = range(1, df.shape[1] + 1)
        print(vif_series[['VIF']])
    return vif_series, df

def get_corr(a, b):
    return pearsonr(a, b)[0]

def get_corr_matrix(df_, target=None, collinear_threshold=0.7,
                    show_response_corr=False, show_feat_name=True,
                    get_vif=False, return_df=False, get_feat_collinearity=True,
                    plt_heatmap=True):
    '''
        df: dataframe of features with/without the response variable
        y: name or series of the response variable
    '''
    df = df_.copy()
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df, columns=range(1, df.shape[1] + 1))
    feat_cols = list(df.columns)

    if target is None:
        starting_feature_index = 0
    else:
        if isinstance(target, str):  # if y is the str-name of the response variable
            feat_cols.remove(target)
        elif isinstance(target, pd.Series):  # if y is the pd.Series of the response variable
            df[target.name] = target.values
            target = target.name
        df = df.loc[:, [target, *feat_cols]].copy()
        starting_feature_index = 1

    feat_dic = dict(zip(range(1, len(feat_cols) + 1), feat_cols))

    def print_feature(i):
        if show_feat_name:
            return '%-20s' % feat_dic[i]
        else:
            return '%-2d' % i

    # Encode object features with label encoder
    object_feature = [i for i, j in zip(df.dtypes.index, df.dtypes) if j == 'object']
    if len(object_feature) > 0:
        for col in object_feature:
            label_encoder = preprocessing.LabelEncoder()
            df.loc[:, col] = label_encoder.fit_transform(df[col]).values

    corr_matrix = df.corr()
    if plt_heatmap:
        # plt.rcParams["figure.figsize"] = [df.shape[1]*2,df.shape[1]*2]
        sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True)
        plt.show()

    corr_matrix = corr_matrix.values
    
    if get_feat_collinearity:
        print('\n\n' + 'Features that have multiColinearity more than %-.2f\n' % collinear_threshold + '-' * 55)
        for i in range(starting_feature_index+1, corr_matrix.shape[0]):
            for j in range(starting_feature_index, i):
                if np.abs(corr_matrix[i, j]) > collinear_threshold:
                    print('%s %s' % (print_feature(i), print_feature(j)), corr_matrix[i, j])

    if (target is not None) and show_response_corr:
        print('=' * 60 + '\n\n' + 'Corr with the Response variable\n' + '-' * 31)
        # tt = df.corr()[target].reset_index()
        # tt.columns = ['Feature_name', 'VIF']
        for j in range(1, corr_matrix.shape[0]):
            print('%s' % print_feature(j), corr_matrix[j, 0])

    if get_vif:
        print('=' * 60, '\n\nVariance Inflation Factor\n' + '-' * 25)
        return get_VIF(df.loc[:, feat_cols].copy(), show_feat_name, return_df)