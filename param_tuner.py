import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
from solver import Solver
from optimizer import Optimizer

import matplotlib
import plt_config
import matplotlib.pyplot as plt
matplotlib.use('Agg') # for running on server without GUI

color = plt.rcParams['axes.prop_cycle'].by_key()['color']

torch.set_default_tensor_type(torch.cuda.FloatTensor
                                    if torch.cuda.is_available()
                                    else torch.FloatTensor)
import json
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


class ParamTuner():
    def __init__(self, batch):
        self.batch = batch
        self.load_space()
        # self.optimizer = Optimizer(batch, V_min=11, V_max=13, Cth_factor=0.5)

    def load_space(self):
        self.space = {
            'V_min': hp.uniform('V_min', 11, 14),
            'V_max': hp.uniform('V_max', 11, 14),
            'Cth_factor': hp.uniform('Cth_factor', 0.1, 0.8)
        }

    def objective(self, param):
        V_min = param['V_min']
        V_max = param['V_max']
        Cth_factor = param['Cth_factor']
        optim = Optimizer(self.batch, V_min, V_max, Cth_factor)
        loss = optim.param_optim_step(V_min, V_max, Cth_factor)
        return {
            'loss': loss,
            'param': param,
            'status': STATUS_OK
        }

    def optimize(self):
        max_evals = 100
        tpe_trials = Trials()
        best = fmin(fn=self.objective, space=self.space, algo=tpe.suggest,
                    max_evals=max_evals, trials=tpe_trials)
        return tpe_trials, best


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor
                                      if torch.cuda.is_available()
                                      else torch.FloatTensor)
    try:
        os.mkdir('results')
    except FileExistsError:
        pass

    batch = 1000
    n_attempts = 10
    tuner = ParamTuner(batch)
    for attempt_id in range(n_attempts):
        tpe_trials, best = tuner.optimize()
        json.dump(tpe_trials.results, open('results/results_{}.json'.format(attempt_id), 'w'))
        json.dump(best, open('results/best_param_{}.json'.format(attempt_id), 'w'))
        print('Attempt {} completed.'.format(attempt_id))
