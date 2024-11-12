import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plt_feature_importances(model, features_names, show_feature_imprtances=1, return_f=False):
    # for more feature importance https://www.kaggle.com/grfiv4/plotting-feature-importances
    feature_import = pd.Series(model.feature_importances_, index=features_names)
    feature_import = feature_import.sort_values(ascending=True)
    if show_feature_imprtances:print( feature_import )
    feature_import.plot(kind='barh', figsize=(7,6))
    plt.savefig('importance.pdf')
    plt.show()
    if return_f:return feature_import
    
def plt_loss(log, metrics_, exp_id, metric_index=0, reporting_=['','val_','test_']):
    try:

        loss = [[log[ep][item + metrics_[metric_index]] for item in reporting_] for ep in log.keys()]
        loss = np.array(loss)

        x = list(log.keys())
        plt.plot(x, loss[:, 0], label='Train')
        plt.plot(x, loss[:, 1], label='Valid')
        if len(reporting_) > 2:
            plt.plot(x, loss[:, 2], label='Test')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.xticks(x, rotation='90')
        plt.legend()
        if exp_id is not None:
            plt.savefig('%s/loss.pdf' % exp_id, bbox_inches='tight')
        plt.show()
        return loss

    except Exception as exc:
        pass

