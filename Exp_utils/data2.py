from .Exp import *

def empty_row(columns, rows=1):
    a = pd.DataFrame(np.array([np.nan] * len(columns)).reshape(1, -1), columns=columns)
    if rows:
        a = a.append(['' for _ in range(rows - 1)])
    return a


cols_with_missing = ['NumberOfSuppliments', 'PastNumberOfClaims', 'Days_Policy_Accident', 'Days_Policy_Claim']

cat_feature = [*cols_with_missing, 'Month', 'DayOfWeek', 'Make', 'AccidentArea', 'DayOfWeekClaimed', 'MonthClaimed', 'Sex',
              'MaritalStatus', 'Fault', 'PolicyType', 'VehicleCategory', 'VehiclePrice', 'Deductible',
              'AgeOfVehicle', 'AgeOfPolicyHolder', 'PoliceReportFiled', 'WitnessPresent', 'AgentType',
              'AddressChange_Claim', 'NumberOfCars', 'Year', 'BasePolicy']

non_cat_feature = ['Age', 'RepNumber', 'WeekOfMonthClaimed', 'WeekOfMonth', 'DriverRating']

cat_feature_without_missing = cat_feature[4:]

Exp_1 = [*cat_feature,*non_cat_feature]
Exp_2 = [*cat_feature,*non_cat_feature]
Exp_3 = [*cat_feature_without_missing,*non_cat_feature, 'Days_Policy_Accident', 'Days_Policy_Claim']
Exp_5 = [*non_cat_feature, *cat_feature[4:]]


def get_data():
    df = pd.read_csv('Dataset-2.csv', skiprows=0, na_values=[' ', '', '  ', '   ', 'none'])
    y = df.pop('FraudFound_P')
    _ = df.pop('PolicyNumber')  # useless: PolicyNumber
    print('df.shape ----------', df.shape)
    return df, y

def get_exp_data(i):
    print('data2: ', 'Exp_%d' % i)
    x, y = get_data()
    if i == 1:
        x = x[Exp_1]
        for f in cat_feature:
            x[f] = LabelEncoder().fit_transform(x[f])
        print(x.shape, 'LE for cat_feature is Done')
        return x, y
    elif i==2:
        x = x[Exp_2]
        for f in cat_feature:
            x[f] = LabelEncoder().fit_transform(x[f])

        for col in cols_with_missing:
            imputed_val = x[col].nunique()-1
            x[col] = x[col].map(lambda x: np.nan if x == imputed_val else x)

        print(x.shape, 'Data with null values, Still Need Data Imputation..')
        return x, y
    elif i==3:
        x = x[Exp_3]
        for f in [*cat_feature_without_missing, 'Days_Policy_Accident', 'Days_Policy_Claim']:
            x[f] = LabelEncoder().fit_transform(x[f])
        print(x.shape, 'LE for cat_feature_without_missing is Done')
        return x, y
    elif i==5:
        x = x[Exp_5]
        for f in x.columns:
            x[f] = LabelEncoder().fit_transform(x[f])

        print(x.shape, 'LE for All_Features are Done')
        return x, y


def get_synthetic_data(n_samples=60, n_features=4, n_classes=2):
    return make_classification(n_classes=n_classes, n_samples=n_samples, n_features=n_features, weights=[0.2], flip_y=0)
