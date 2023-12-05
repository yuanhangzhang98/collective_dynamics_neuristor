import os
import numpy as np
import torch
from tqdm import trange
from solver import Solver

torch.set_default_tensor_type(torch.cuda.FloatTensor
                                    if torch.cuda.is_available()
                                    else torch.FloatTensor)

try:
    os.mkdir('results')
except FileExistsError:
    pass

try:
    os.mkdir('peaks')
except FileExistsError:
    pass

try:
    os.mkdir('avalanches')
except FileExistsError:
    pass

try:
    os.mkdir('graphs')
except FileExistsError:
    pass

# Ns = [1024, 4096, 16384, 65536]
Ns = [4096]

# V_batch = np.linspace(13.444, 13.476, 9)
V_batch = [9.96, 13.44]
Cth_batch = np.ones_like(V_batch)

for batch_idx in range(len(V_batch)):
    V_batch_i = V_batch[batch_idx]
    Cth_batch_i = Cth_batch[batch_idx]

    for N in Ns:
        # minibatch = int(131072 / N)
        minibatch = int(524288 / N)
        d = 2
        Vs = [V_batch_i] * minibatch
        R = 12
        noise_strength = 0.001
        Cth_factors = [Cth_batch_i]
        couple_factor = 0.02
        width_factor = 1.0
        T_base = 325
        t_max = int(1e6)
        dt = 10

        min_dist = 101
        if min_dist % 2 == 0:
            min_dist += 1
        peak_threshold = 1.5

        len_x = 1
        len_y = 40

        Vs, Cth_factors = np.meshgrid(Vs, Cth_factors)
        Vs = torch.tensor(Vs.reshape(-1)).reshape(-1, 1)
        Cth_factors = torch.tensor(Cth_factors.reshape(-1)).reshape(-1, 1)
        batch = len(Vs)

        n_minibatch = int(np.ceil(batch / minibatch))

        for i in trange(n_minibatch):
            Vi = Vs[i * minibatch: (i + 1) * minibatch]
            Cth_factori = Cth_factors[i * minibatch: (i + 1) * minibatch]

            V = Vi[0, 0]
            Cth_factor = Cth_factori[0, 0]
            n_repeat = 100

            solver = Solver(d, len(Vi), N, V, R, noise_strength, Cth_factor, couple_factor, width_factor,
                            T_base, t_max, dt, n_repeat, peak_threshold, min_dist, len_x, len_y)
            solver.model.set_input(Vi, Cth_factori)
            solver.run(ensemble=True)

