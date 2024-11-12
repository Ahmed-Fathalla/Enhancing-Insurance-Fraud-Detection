import sys
import numpy as np
import pandas as pd
import warnings;warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import operator
from tabulate import tabulate

def sort_tuble(tub, item = 2, ascending = True, return_top=False):
    tub = sorted(tub, key=operator.itemgetter(item), reverse=ascending)
    if return_top:
        if ascending:
            print( tub[0])
        else:
            print( tub[-1])
    return tub

def train_test_split(x_data_, y_data_, train_ratio = 0.4):
    train_obs = int(x_data_.shape[0]*train_ratio)
    X_train, y_train = x_data_[:train_obs], y_data_[:train_obs]
    X_valid, y_valid = x_data_[train_obs:], y_data_[train_obs:]
    return X_train, y_train, X_valid, y_valid

def df_to_file(df, cols=None, fold_col=None, round_=5, save_to_file=None, padding='left', rep_newlines='\t',
               print_=True, wide_col='', pre='', post=''):
    if type(df) is list:
        df = pd.DataFrame(np.array(df), columns=cols)
    elif type(df) is pd.DataFrame:
        ...


    headers = [wide_col+str(i)+wide_col for i in df.columns.values]

    df = df.round(round_)
    if fold_col is not None:
        df['fold'] = fold_col
        df = df[['fold']+headers]
    else:
        df = df[headers]

    c = rep_newlines + tabulate(df.values,
                                headers=headers,
                                stralign=padding,
                                disable_numparse=1,
                                tablefmt = 'grid' # 'fancy_grid' ,
                                ).replace('\n', '\n'+rep_newlines)
    if print_:print(c)
    if save_to_file is not None:
        with open(save_to_file, 'a', encoding="utf-8") as myfile:
            myfile.write( pre + c + post + '\n')
    return df

import matplotlib.pyplot as plt
def plt_loss(file, hist, metric):
    from .pkl_utils import dump
    import traceback
    try:
        dump('%s.pkl' % file, hist)
    except Exception as exc:
        pass
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_'+metric])
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], )
    plt.savefig('%s.pdf'%file, bbox_inches='tight')
    plt.show()