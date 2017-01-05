import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class ModelOptimize:
    def __init__(self, skmodel, params_opt_dic, X, y, val_metrics, val_size=0.3):
        """
        :param skmodel: sklearn model to evaluate
        :param params_opt_dic: dictionary with parameters to optimize and ranges
        :param X: Data
        :param y: Target
        :param val_size: part of data for validation
        :param val_metrics: validation metric. An object from sklearn.metrics

        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size)

        self.skmodel = skmodel
        self.params = params_opt_dic
        self.X = X_train
        self.y = y_train
        self.val_X = X_val
        self.val_y = y_val
        self.val_metrics = val_metrics

    def rand_config(self):
        """ Returns random configuration """
        rand_conf = {}
        for key, val in self.params.items():
            if type(val[0]) == str:
                rand_conf[key] = np.random.choice(val)
                continue

            rand_val = np.random.uniform()
            if len(val) == 2:
                length = val[1] - val[0]
                if type(val[1]) is int:
                    rand_conf[key] = int(val[0] + length * rand_val)
                else:
                    rand_conf[key] = val[0] + length * rand_val
            elif len(val) == 1:
                rand_conf[key] = val[0]
                continue
            else:
                raise Exception("Say range for parameters or single value, not grid")
        return rand_conf

    def config2loss(self, iters, config):
        """ Evaluates model trained through `iters` iteration
         with given configuration on validation data """

        fit_params = {'nb_epoch': int(iters), 'verbose': 0}
        self.skmodel.set_params(**config)
        self.skmodel.fit(self.X, self.y, **fit_params)
        result = self.skmodel.predict(self.val_X)
        return self.val_metrics(self.val_y, result)
