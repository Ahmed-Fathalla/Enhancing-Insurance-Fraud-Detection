import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

import warnings, sys
warnings.filterwarnings('ignore')

'''
dataframe_preprocess(df)
dataframe_reduce_memory_usage(df)

get_skewness
get_nulls
'''

def display(dic):
    if isinstance(dic, pd.Series):dic=dic.values
    if isinstance(dic, np.ndarray):dic=dic.flatten().tolist()
    if isinstance(dic, list):dic = Counter(dic)
    print('\t%-15s %-5s\n\t%s'%('value','| Count','-'*16 + '|-----'  ))
    keys = sorted(dic.keys())
    for k in keys:
        print('\t%-15s | %-5d'%(k,dic[k]))

# def display(dic):
#     if isinstance(dic, pd.Series):dic=dic.values
#     if isinstance(dic, np.ndarray):dic=dic.flatten().tolist()
#     if isinstance(dic, list):dic = Counter(dic)
#     print('%-15s %-5s\n%s'%('value','| Count','-'*16 + '|-----'  ))
#     keys = sorted(dic.keys())
#     for k in keys:
#         print('%-15s | %-5d'%(k,dic[k]))

def get_categorical_features(x,thresholds_ratio=0.05, verboseunique=0):
    df_len = x.shape[0]
    categorical_features = []
    for i in x.columns:
        unique = x[i].nunique()
        if unique/df_len < thresholds_ratio:
            categorical_features.append([i,unique])
            if verboseunique: print('%-20s'%i,x[i].nunique() )
    return np.array(categorical_features) 
    
def hello_df(df, thresholds_ratio=1.1, save_plt=False, show_plt=True, verbose=1):
    '''
    parameters:
        df: DataFrame to display its data
    just like .describe(), but 
        1. display the results for categorical variables only. 
        2. return the cols that have 'object' dtypes
    '''
    max_len = 0
    for i in df.columns.values:
        if len(i)> max_len:
            max_len=len(i)
    
    df = df.copy()
    print( 'Original Data:      ', df.shape )
    print( 'Without duplication:' , df.drop_duplicates().shape )
    categorical_features = get_categorical_features(df,thresholds_ratio=thresholds_ratio)
    # print(categorical_features)
    if verbose:
        print( '='*(max_len+35) ,'\n%s\t'%pdng('Col_Name',max_len),'      DataType',' NAN     %')
        print('='*(max_len+35))
    
    for i,j in zip(df.dtypes.index,df.dtypes):
        s = ''
        # if j == 'object':s = '%-15s----- '%i
        if i in categorical_features[:,0]:
            s = '%s  %-5d--- '%(pdng(i,max_len+1),df[i].nunique())
        else:
            s='%s           '%pdng(i,max_len+1) 
        
        s += '%-10s'%j
        s += '%-5d(%-5s%%)'%(df[i].isnull().sum(), str(df[i].isnull().sum() * 100/ len(df))[:5])
        if verbose:print(s)
        
    continuous_features = [i for i in df.columns if i not in categorical_features]
    
    if verbose:print('='*30,'\n'*2)
    
    null_percent = get_nulls(df, save_plt=save_plt, show_plt=show_plt, verbose=verbose)

    # from IPython.display import display, HTML
    # display(HTML(df[df[categorical_features[:,0]]].describe().to_html()))
    
    return categorical_features, continuous_features, null_percent # list(df.select_dtypes(include=['category','object']))

def get_nulls(df, save_plt=False, show_plt=True, verbose=1):
    if df.isnull().sum().values.sum()==0:  # means that no null values.
        return [],[]
    df_train_na = (df.isnull().sum() / len(df)) * 100.0    
    df_train_na = df_train_na.drop(df_train_na[df_train_na == 0].index).sort_values(ascending=False)[:30]
    df_train_na = pd.DataFrame({'Missing Ratio' :df_train_na})
    df_train_na = df_train_na.reset_index()
    if verbose:
        print('No of colums with nulls: ',df_train_na.shape[0],'\n', '='*28, sep='')
    null_percent = []
    null_cols = df_train_na['index'].values
    for i in df_train_na.values:
        i = list(i)
        # i = str(i).split(' ')
        null_percent.append(round(i[1],3))
    
    null_percent = np.array(list(zip(null_cols, null_percent)))
    
    if verbose:
        print( df_train_na)#.head(10) )  
        
    if verbose:print('\n'*2)
    
    # if save_plt or show_plt:
    #     sys.path.append("..")
    #     from Plots.plot import plt_nulls, plt_null_indeces
    #     plt_nulls(df_train_na, save_plt=save_plt, show_plt=show_plt)
    #     plt_null_indeces(df[null_cols], save_plt=save_plt, show_plt=show_plt)

    return null_percent

def printall_columns(x,max_rows=10):
    from IPython.display import display, HTML
    display(HTML(x.to_html(max_rows=max_rows)))   
    
def consider_NaN(df, col=None, Nan_str='NaN****'):
    '''
    makes nan values as string of 'NaN'
    '''
    if col is  None:
        col = df.columns
    for i in col:
        df.ix[df[i].isnull(),i]  = Nan_str
    return df
                    
def concat_df_vertically():
    dd = pd.read_csv('csv/cars  n_5 Model:__LinearSVR__    outlier=None   len:3510432019_07_10_5_26_PM.csv',delimiter=',')
    # dd.drop('avg_str', axis=1, inplace = False)
    features = pd.concat([ dd[['y_pred_transform','avg']] ,  
                           pd.get_dummies(dd['avg'].apply(lambda x:str(x)), prefix='avg')], 
                         axis=1)
                         
def pvtbl(df):
    '''
    https://github.com/nicolaskruchten/pivottable
    https://github.com/nicolaskruchten/pyconca/blob/master/jupyter_magic.ipynb
    test online: https://pivottable.js.org/examples/rcsvs.html
    youtube toutorials: https://www.youtube.com/watch?v=MVrlFs3TTcQ
    '''
    from pivottablejs import pivot_ui
    pivot_ui(df)
    
def indent_ds(ds, sp=6):
    ds.index = [' '*sp+str(i) for i in ds.index ]
    return ds

def pdng(str_, i, padding_char = ' '):
    return str_+ padding_char*(i-len(str_))
	
def read_df(path):
    df = pd.read_csv(path)
    return df    

