from Exp_utils.data2 import *

df, y = get_exp_data(1)
run_main(df,y, exp_str='exp_1')

df, y = get_exp_data(2)
for imp in ['iterative','knn', 'simple']:
    x = imputer_(df, cols_with_missing, method=imp)
    run_main(x, y, exp_str='exp_2_'+imp)

df, y = get_exp_data(3)
run_main(df,y, exp_str='exp_3')

df, y = get_exp_data(5)
run_main(df,y, exp_str='exp_5')