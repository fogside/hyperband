"""
    hyperband.py
    
    `hyperband` algorithm for hyper-parameter optimization
    
    Copied/adapted from https://people.eecs.berkeley.edu/~kjamieson/hyperband.html
    
    The algorithm is motivated by the idea that random search is pretty 
    good as far as black-box optimization goes, so let's try to do it faster.
"""

from math import log, ceil
from model_optimize import ModelOptimize


class HyperBand:
    def __init__(self, model, max_iter=81, eta=3, max_metric=True):

        """
        :param model: object of class ModelOptimize
        :param max_iter: max iterations to run with one estimator
        :param eta: Manage num of iterations per one algo and how many algos to save
                    (recommended use default=3)
        :param max_metric: does the process maximize metric? (Default: True)

        """

        self.model = model
        self.max_iter = max_iter
        self.eta = eta
        self.s_max = int(log(max_iter) / log(eta))
        self.B = (self.s_max + 1) * max_iter
        self.max_metric = max_metric
        self.history = []
        self.total_iters = 0
        self.best = []

        print("self.s_max", self.s_max)

    def run(self):
        for s in reversed(range(self.s_max + 1)):
            print('s: ', s)

            # initial number of configs
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # initial configs
            configs = [self.model.rand_config() for _ in range(n)]
            for i in range(s + 1):
                r_i = r * self.eta ** i

                val_losses = []
                for config in configs:
                    self.total_iters += r_i
                    print('config', config)
                    val_loss = self.model.config2loss(iters=r_i, config=config)
                    print("Loss: %f" % val_loss)
                    val_losses.append(val_loss)

                these_results = zip(configs, val_losses, [r_i] * len(configs))
                if self.max_metric:
                    these_results = sorted(these_results, key=lambda x: x[1], reverse=True)
                else:
                    these_results = sorted(these_results, key=lambda x: x[1])
                self.history += these_results

                n_keep = int(n * self.eta ** (-i - 1))
                configs = [config for config, loss, iters in these_results[:n_keep]]
            self.best.append(self.history[-1])


def run_hyperband(skmodel, params_opt_dic, X, y, val_metrics):
    """
    :param skmodel: model with sklearn interface: required fit/predict methods
    :param params_opt_dic:
    :param X: data
    :param y: target
    :param val_metrics: object from sklearn.metrics
    :return: list of best params

    """

    model = ModelOptimize(skmodel, params_opt_dic, X, y, val_metrics=val_metrics)
    hb = HyperBand(model)
    hb.run()
    print("DONE\n\n\n")
    print("hb.total_iters:", hb.total_iters)
    print('--------------------------------')
    print(hb.best)
    return hb.best
