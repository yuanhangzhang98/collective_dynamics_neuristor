import os
import numpy as np
import torch
from tqdm import trange
from solver import Solver
from KS_equation import dynamics
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


torch.set_default_tensor_type(torch.cuda.FloatTensor
                                    if torch.cuda.is_available()
                                    else torch.FloatTensor)

os.makedirs('results', exist_ok=True)
os.makedirs('graphs', exist_ok=True)
os.makedirs('avalanches', exist_ok=True)

Ns = [256]

for N in Ns:
    # minibatch = int(131072 / N)
    # minibatch = int(262144 / N)
    minibatch = 1
    minibatch_repeat = 1
    L = int(np.sqrt(N))
    u0 = torch.randn(minibatch, L, L)
    u0 = u0 - u0.mean(dim=(1, 2), keepdim=True)
    dt = 0.05
    n_steps = 1000
    trajectory = torch.load('ckpts/test_traj.pt')
    # trajectory = dynamics(u0, dt, n_steps)
    trajectory_normalized = ((trajectory - trajectory.mean(dim=(1, 2), keepdim=True) + 10) / 20).clamp(0, 1)

    fig, ax = plt.subplots()
    im1 = ax.imshow(trajectory_normalized[0].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im1)
    plt.savefig('graphs/trajectory.png', dpi=300)
    plt.close()

    # V_min = 12.413381282561543
    # V_max = 12.19013679073254
    # Cth_factor = 0.43684498014118855
    V_min = 11.82557193308969
    V_max = 11.348429087250674
    Cth_factor = 1.0730633816113209
    V_in = (trajectory_normalized[0] * (V_max - V_min) + V_min).reshape(minibatch, N)
    d = 2
    R = 12
    noise_strength = 0
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

    Vi = V_in
    Cth_factori = Cth_factor * torch.ones(minibatch, N)

    V = Vi[0, 0]
    Cth_factor = Cth_factori[0, 0]
    n_repeat = 100

    for batch_idx in trange(minibatch_repeat):
        solver = Solver(d, len(Vi), N, V, R, noise_strength, Cth_factor, couple_factor, width_factor,
                        T_base, t_max, dt, n_repeat, peak_threshold, min_dist, len_x, len_y, batch_idx)
        solver.model.set_input(Vi, Cth_factori)
        solver.run(plot_2D=True, ensemble=True)

