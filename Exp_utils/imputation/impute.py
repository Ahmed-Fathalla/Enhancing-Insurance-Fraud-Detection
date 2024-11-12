from sklearn.experimental import enable_iterative_imputer

import pandas as pd

from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

# https://chat.openai.com/share/b1b558fe-fc64-4056-a27a-4f5810ce617c
# import numpy as np
# data = {'Category': ['A', np.nan, 'B', 'C', 'B', np.nan, 'A', 'C']}
# df = pd.DataFrame(data)
# def random_impute(df, column_name):
#     # Get a list of non-missing values
#     non_missing = df[column_name].dropna().values
#     # Randomly fill missing values with values from non-missing data
#     df[column_name] = df[column_name].apply(lambda x: np.random.choice(non_missing) if pd.isnull(x) else x)
#     return df
# df_imputed = random_impute(df, 'Category')
# print(df_imputed)

def imputer_(df, cols_with_missing, method=''):

    if method == 'iterative':
        im = IterativeImputer(random_state=100)
        print('Imputation method: IterativeImputer')
    elif method == 'knn':
        # https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html#sklearn.impute.KNNImputer
        im = KNNImputer(n_neighbors=5)
        print('Imputation method: KNNImputer')
    elif method == 'simple':
        # https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer
        im = SimpleImputer(strategy='median')
        print('Imputation method: SimpleImputer(strategy="median")')
    else:
        print("invalid method, pls select one of the following ['iterative','knn', 'simple']")
        return None

    cols = df.columns.values  # [['months_as_customer',*cols_with_missing,*cols_without_missing]]
    im_df = im.fit_transform(df.values)

    im_df = pd.DataFrame(im_df, columns=cols)

    for col in cols_with_missing:
        s = im_df[col].nunique()
        im_df[col] = im_df[col].round()
        print('\t', col, "unique:", s, '=>', im_df[col].nunique())

    return im_df





#
# import numpy as np
# import torch
# from geomloss import SamplesLoss
# # from ..utils.ot_utils import *
# from utils.ot_utils import *
#
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer
#
#
#
# class BatchSinkhornImputation():
#     """
#     'One parameter equals one imputed value' model (Algorithm 1. in the paper)
#     Parameters
#     ----------
#     eps: float, default=0.01
#         Sinkhorn regularization parameter.
#
#     lr : float, default = 0.01
#         Learning rate.
#     opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
#         Optimizer class to use for fitting.
#
#     max_iter : int, default=10
#         Maximum number of round-robin cycles for imputation.
#     niter : int, default=15
#         Number of gradient updates for each model within a cycle.
#     batchsize : int, defatul=128
#         Size of the batches on which the sinkhorn divergence is evaluated.
#     n_pairs : int, default=10
#         Number of batch pairs used per gradient update.
#     tol : float, default = 0.001
#         Tolerance threshold for the stopping criterion.
#     weight_decay : float, default = 1e-5
#         L2 regularization magnitude.
#     order : str, default="random"
#         Order in which the variables are imputed.
#         Valid values: {"random" or "increasing"}.
#     unsymmetrize: bool, default=True
#         If True, sample one batch with no missing
#         data in each pair during training.
#     scaling: float, default=0.9
#         Scaling parameter in Sinkhorn iterations
#         c.f. geomloss' doc: "Allows you to specify the trade-off between
#         speed (scaling < .4) and accuracy (scaling > .9)"
#     """
#
#     def __init__(self,
#                  eps=0.01,
#                  lr=1e-2,
#                  opt=torch.optim.RMSprop,
#                  niter=2000,
#                  batchsize=32,
#                  n_pairs=1,
#                  noise=0.1,
#                  scaling=.9):
#         self.eps = eps
#         self.lr = lr
#         self.opt = opt
#         self.niter = niter
#         self.batchsize = batchsize
#         self.n_pairs = n_pairs
#         self.noise = noise
#         self.sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=scaling, backend="tensorized")
#
#     def fit_transform(self, X, verbose=True, report_interval=500, X_true=None):
#         """
#         Imputes missing values using a batched OT loss
#         Parameters
#         ----------
#         X : torch.DoubleTensor or torch.cuda.DoubleTensor
#             Contains non-missing and missing data at the indices given by the
#             "mask" argument. Missing values can be arbitrarily assigned
#             (e.g. with NaNs).
#         mask : torch.DoubleTensor or torch.cuda.DoubleTensor
#             mask[i,j] == 1 if X[i,j] is missing, else mask[i,j] == 0.
#         verbose: bool, default=True
#             If True, output loss to log during iterations.
#         X_true: torch.DoubleTensor or None, default=None
#             Ground truth for the missing values. If provided, will output a
#             validation score during training, and return score arrays.
#             For validation/debugging only.
#         Returns
#         -------
#         X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
#             Imputed missing data (plus unchanged non-missing data).
#         """
#
#         X = torch.tensor(torch.from_numpy(X).clone().detach())
#         n, d = X.shape
#
#         mask = torch.isnan(X).double()
#         imps = (self.noise * torch.randn(mask.shape).double() + nanmean(X, 0))[mask.bool()]
#         imps.requires_grad = True
#         optimizer = self.opt([imps], lr=self.lr)
#
#         if X_true is not None:
#             maes = np.zeros(self.niter)
#             rmses = np.zeros(self.niter)
#
#         for i in range(self.niter):
#
#             X_filled = X.detach().clone()
#             X_filled[mask.bool()] = imps
#             loss = 0
#
#             for _ in range(self.n_pairs):
#                 idx1 = np.random.choice(n, self.batchsize, replace=False)
#                 idx2 = np.random.choice(n, self.batchsize, replace=False)
#
#                 X1 = X_filled[idx1]
#                 X2 = X_filled[idx2]
#
#                 loss = loss + self.sk(X1, X2)
#
#             if torch.isnan(loss).any() or torch.isinf(loss).any():
#                 ### Catch numerical errors/overflows (should not happen)
#                 print("Nan or inf loss\n")
#                 break
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         X_filled = X.detach().clone()
#         X_filled[mask.bool()] = imps
#
#         return X_filled
#
#
# class RRimputer():
#     """
#     Round-Robin imputer with a batch sinkhorn loss
#     Parameters
#     ----------
#     models: iterable
#         iterable of torch.nn.Module. The j-th model is used to predict the j-th
#         variable using all others.
#     eps: float, default=0.01
#         Sinkhorn regularization parameter.
#
#     lr : float, default = 0.01
#         Learning rate.
#     opt: torch.nn.optim.Optimizer, default=torch.optim.Adam
#         Optimizer class to use for fitting.
#
#     max_iter : int, default=10
#         Maximum number of round-robin cycles for imputation.
#     niter : int, default=15
#         Number of gradient updates for each model within a cycle.
#     batchsize : int, defatul=128
#         Size of the batches on which the sinkhorn divergence is evaluated.
#     n_pairs : int, default=10
#         Number of batch pairs used per gradient update.
#     tol : float, default = 0.001
#         Tolerance threshold for the stopping criterion.
#     weight_decay : float, default = 1e-5
#         L2 regularization magnitude.
#     order : str, default="random"
#         Order in which the variables are imputed.
#         Valid values: {"random" or "increasing"}.
#     unsymmetrize: bool, default=True
#         If True, sample one batch with no missing
#         data in each pair during training.
#     scaling: float, default=0.9
#         Scaling parameter in Sinkhorn iterations
#         c.f. geomloss' doc: "Allows you to specify the trade-off between
#         speed (scaling < .4) and accuracy (scaling > .9)"
#     """
#
#     def __init__(self,
#                  models,
#                  eps=0.01,
#                  lr=1e-2,
#                  opt=torch.optim.Adam,
#                  max_iter=1,
#                  niter=1,
#                  batchsize=4,
#                  n_pairs=10,
#                  tol=1e-3,
#                  noise=0.1,
#                  weight_decay=1e-5,
#                  order='random',
#                  unsymmetrize=True,
#                  scaling=.9):
#
#         self.models = models
#         self.sk = SamplesLoss("sinkhorn", p=2, blur=eps,
#                               scaling=scaling, backend="auto")
#         self.lr = lr
#         self.opt = opt
#         self.max_iter = max_iter
#         self.niter = niter
#         self.batchsize = batchsize
#         self.n_pairs = n_pairs
#         self.tol = tol
#         self.noise = noise
#         self.weight_decay = weight_decay
#         self.order = order
#         self.unsymmetrize = unsymmetrize
#
#         self.is_fitted = False
#
#     def fit_transform(self, X, verbose=True,
#                       report_interval=1, X_true=None):
#         """
#         Fits the imputer on a dataset with missing data, and returns the
#         imputations.
#         Parameters
#         ----------
#         X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
#             Contains non-missing and missing data at the indices given by the
#             "mask" argument. Missing values can be arbitrarily assigned
#             (e.g. with NaNs).
#         mask : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
#             mask[i,j] == 1 if X[i,j] is missing, else mask[i,j] == 0.
#         verbose : bool, default=True
#             If True, output loss to log during iterations.
#
#         report_interval : int, default=1
#             Interval between loss reports (if verbose).
#         X_true: torch.DoubleTensor or None, default=None
#             Ground truth for the missing values. If provided, will output a
#             validation score during training. For debugging only.
#         Returns
#         -------
#         X_filled: torch.DoubleTensor or torch.cuda.DoubleTensor
#             Imputed missing data (plus unchanged non-missing data).
#         """
#
#         X = torch.from_numpy(X).clone().to('cuda')
#         n, d = X.shape
#         mask = torch.isnan(X).double().to('cuda')
#         normalized_tol = self.tol * torch.max(torch.abs(X[~mask.bool()]))
#         order_ = torch.argsort(mask.sum(0))
#
#         optimizers = [self.opt(self.models[i].parameters(),
#                                lr=self.lr, weight_decay=self.weight_decay) for i in range(d)]
#
#         imps = (self.noise * torch.randn(mask.shape).double().to('cuda') + nanmean(X, 0).to('cuda'))[mask.bool()]
#         X[mask.bool()] = imps
#         X_filled = X.clone()
#
#         for i in range(self.max_iter):
#
#             if self.order == 'random':
#                 order_ = np.random.choice(d, d, replace=False)
#             X_old = X_filled.clone().detach()
#
#             loss = 0
#
#             for l in range(d):
#                 j = order_[l].item()
#                 n_not_miss = (~mask[:, j].bool()).sum().item()
#
#                 if n - n_not_miss == 0:
#                     continue  # no missing value on that coordinate
#
#                 for k in range(self.niter):
#
#                     loss = 0
#
#                     X_filled = X_filled.detach()
#                     X_filled[mask[:, j].bool(), j] = self.models[j](
#                         X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j + 1: d]]).squeeze()
#
#                     for _ in range(self.n_pairs):
#
#                         idx1 = np.random.choice(n, self.batchsize, replace=False)
#                         X1 = X_filled[idx1]
#
#                         if self.unsymmetrize:
#                             n_miss = (~mask[:, j].bool()).sum().item()
#                             idx2 = np.random.choice(n_miss, self.batchsize, replace=self.batchsize > n_miss)
#                             X2 = X_filled[~mask[:, j].bool(), :][idx2]
#
#                         else:
#                             idx2 = np.random.choice(n, self.batchsize, replace=False)
#                             X2 = X_filled[idx2]
#
#                         loss = loss + self.sk(X1, X2)
#
#                     optimizers[j].zero_grad()
#                     loss.backward()
#                     optimizers[j].step()
#
#                 # Impute with last parameters
#                 with torch.no_grad():
#                     X_filled[mask[:, j].bool(), j] = self.models[j](
#                         X_filled[mask[:, j].bool(), :][:, np.r_[0:j, j + 1: d]]).squeeze()
#
#         self.is_fitted = True
#         return X_filled