# from sklearn.linear_model import Lasso
# from sklearn.linear_model import Elast
# import catboost # boosting Algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier

class stacking_:
    def __init__(self, estimators, final_estimator, cv=10):
        self.estimators = estimators
        self.final_estimators = final_estimator
        self.m = StackingClassifier(estimators=estimators,
                                    final_estimator=final_estimator,
                                    cv=cv)

    def fit(self, X_train, y_train):
        self.m.fit(X_train, y_train)

    def predict(self, X_test):
        return self.m.predict(X_test)


# def stacking(base_estimators, final_estimators, X_train, X_test, y_train, y_test, scoring_method, cv=10):
#     for i, m in base_estimators:
#         m.fit(X_train, y_train)
#         y_trn_pred = m.predict(X_train)
#         y_tst_pred = m.predict(X_test)
#         scoring_method(y_train, y_trn_pred)
#         scoring_method(y_test, y_tst_pred)         # print("%s training Accuracy:" % i, m.score(X_train, y_train))     # print("%s test Accuracy:" % i, m.score(X_test, y_test), '\n')
#
#     # ===========================================================================================
#     stacking = StackingClassifier(estimators=base_estimators,
#                               final_estimator=final_estimators,
#                               cv=cv)
#
#     stacking.fit(X_train, y_train)
#     y_trn_pred = stacking.predict(X_train)
#     y_tst_pred = stacking.predict(X_test)
#     scoring_method(y_train, y_trn_pred)
#     scoring_method(y_test, y_tst_pred)    # print("Stacking training Accuracy:", sclf.score(X_train, y_train))    # print("Stacking test Accuracy:", sclf.score(X_test, y_test))

cat_feature = ['insured_sex', 'insured_education_level', 'insured_occupation', 'incident_severity',
            'incident_type', 'authorities_contacted', 'bodily_injuries', 'witnesses',]
            # 'property_damage', 'police_report_available', 'collision_type']
        
models = [BaggingClassifier( random_state = 123),
          RandomForestClassifier(random_state = 123),
          LogisticRegression(random_state = 123),
          RidgeClassifier(random_state = 123),
          AdaBoostClassifier(random_state = 123),
          XGBClassifier(random_state = 123),         # catboost.CatBoostClassifier( verbose=0, cat_features=cat_feature),
          GaussianNB(),
          LinearSVC(random_state = 123),
          KNeighborsClassifier(),
          StackingClassifier( estimators= [('rf',           RandomForestClassifier(n_estimators=100, random_state=123)),
                                           ('logistic_reg', LogisticRegression(random_state=123)),
                                           ('KNN',          KNeighborsClassifier())],
                              final_estimator = XGBClassifier(random_state=123, verbosity=0, use_label_encoder=False),
                              cv=10)
         ]


#
# # from utils.scoring import get_res
# # from utils.time_utils import get_TimeStamp_str
# # from sklearn.model_selection import train_test_split
# import catboost
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LogisticRegression
# # from sklearn.linear_model import Lasso
# # from sklearn.linear_model import Ridge
# # from sklearn.linear_model import ElasticNet
# from sklearn.ensemble import BaggingRegressor
# # from sklearn.ensemble import AdaBoostRegressor
# from sklearn.svm import LinearSVC
# from ..data import cat_features
#
# from sklearn.neighbors import KNeighborsClassifier
# # import catboost
#
# model_lst = [
#              RandomForestRegressor(),
#              catboost.CatBoostRegressor(iterations=1000, verbose=0, cat_features=cat_features),
#              BaggingRegressor(),
#              LogisticRegression(),
#              KNeighborsClassifier(),
#              LinearSVC(),
#              GaussianNB(),
#             #  Lasso(),
#             # Ridge(),
#             # ElasticNet(),
#             # AdaBoostRegressor()
#              ]
#
