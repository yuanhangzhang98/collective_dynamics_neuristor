import os
import numpy as np
import torch
from tqdm import trange
from solver import Solver

torch.set_default_tensor_type(torch.cuda.FloatTensor
                                    if torch.cuda.is_available()
                                    else torch.FloatTensor)

os.makedirs('results', exist_ok=True)
os.makedirs('avalanches', exist_ok=True)
os.makedirs('graphs', exist_ok=True)
os.makedirs('fits', exist_ok=True)
os.makedirs('peaks', exist_ok=True)


Ns = [4096]
V = 9.96
Cth_factor = 1.0

for N in Ns:
    minibatch = int(262144 / N)
    minibatch_repeat = int(N / 1024)
    d = 2
    Vs = V * np.ones(minibatch)
    R = 12
    noise_strength = 0.001
    Cth_factors = Cth_factor * np.ones(1)
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
        n_repeat = 100
        for batch_idx in trange(minibatch_repeat):
            solver = Solver(d, len(Vi), N, V, R, noise_strength, Cth_factor, couple_factor, width_factor,
                            T_base, t_max, dt, n_repeat, peak_threshold, min_dist, len_x, len_y, batch_idx)
            solver.model.set_input(Vi, Cth_factori)
            solver.run(ensemble=True, plot_2D=True)

