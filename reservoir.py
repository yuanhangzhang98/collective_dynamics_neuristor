import numpy as np
import torch
import torch.nn as nn
from model import Circuit, Circuit2D
from utils import find_peaks, bin_traj


class Reservoir2D(nn.Module):
    def __init__(self, batch, Nx, Ny, N_out, V_min=11, V_max=13, Cth_factor=1.0, noise_strength=0.001):
        super().__init__()
        self.batch = batch
        self.Nx = Nx
        self.Ny = Ny
        self.N = self.Nx * self.Ny
        self.N_out = N_out
        self.V = V_min
        self.V_min = V_min
        self.V_max = V_max
        self.R = 12
        self.noise_strength = noise_strength
        self.Cth_factor = Cth_factor
        self.couple_factor = 0.02
        self.width_factor = 1.0
        self.T_base = 325
        self.t_max = 10000
        self.dt = 10
        self.n_repeat = 1
        self.peak_threshold = 1.5
        self.min_dist = 101
        self.save_length = 1000
        self.transient = 0.5
        self.len_x = 1
        self.len_y = 50

        self.n_step = int(np.ceil(self.t_max / self.dt))
        self.len_t = int(np.ceil(self.n_step / self.len_y))

        self.name_string = f'{self.N}_{self.V:.1f}_{1000 * self.noise_strength:.4f}_{self.Cth_factor:.4f}_' \
                           f'{self.couple_factor:.4f}_{self.width_factor:.4f}'

        self.reservoir = Circuit2D(self.batch, self.Nx, self.Ny, self.V, self.R, self.noise_strength, self.Cth_factor,
                                   self.couple_factor, self.width_factor, self.T_base)
        self.out = nn.Linear(self.N * self.len_t, self.N_out)

    def reset(self, V_min, V_max, Cth_factor, noise_strength):
        self.V_min = V_min
        self.V_max = V_max
        self.Cth_factor = Cth_factor
        self.noise_strength = noise_strength
        self.reservoir.__init__(self.batch, self.Nx, self.Ny, self.V, self.R, self.noise_strength, self.Cth_factor,
                                self.couple_factor, self.width_factor, self.T_base)
        self.out.reset_parameters()

    def reservoir_func(self, V_input):
        self.reservoir.set_input(V=V_input)
        y = torch.stack([torch.zeros(self.batch, self.N), torch.ones(self.batch, self.N) * self.T_base], dim=1)
        y, I_traj = self.reservoir.solve(y, self.t_max, self.dt)
        peaks = find_peaks(I_traj.reshape(self.batch * self.N, self.n_step), self.peak_threshold, self.min_dist)
        peaks = peaks.reshape(self.batch, self.N, self.n_step)
        binned_traj = bin_traj(peaks, self.len_x, self.len_y)  # (batch, N, len_t)
        return binned_traj.reshape(self.batch, self.N * self.len_t)

    def forward(self, x):
        x = self.V_min + (self.V_max - self.V_min) * x
        reservoir_output = self.reservoir_func(x)
        out = self.out(reservoir_output)
        return nn.functional.log_softmax(out, dim=-1)


