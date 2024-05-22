import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
from model import Circuit2D


class Reservoir2D:
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
        # self.out = nn.Linear(self.N * self.len_t, self.N_out)
        self.W = None

    def reset(self, V_min, V_max, Cth_factor, noise_strength):
        self.V_min = V_min
        self.V_max = V_max
        self.Cth_factor = Cth_factor
        self.noise_strength = noise_strength
        self.reservoir.__init__(self.batch, self.Nx, self.Ny, self.V, self.R, self.noise_strength, self.Cth_factor,
                                self.couple_factor, self.width_factor, self.T_base)
        # self.out.reset_parameters()

    def reservoir_func(self, V_input):
        n_minibatch = int(np.ceil(V_input.shape[0] / self.batch))
        out = []
        print('Simulating reservoir dynamics......')
        for i in trange(n_minibatch):
            V_input_i = V_input[i * self.batch: (i + 1) * self.batch]
            if len(V_input_i) < self.batch:
                V_input_i = torch.cat([V_input_i, torch.zeros(self.batch - len(V_input_i), self.N)], dim=0)
            self.reservoir.set_input(V=V_input_i)
            y = torch.stack([torch.zeros(self.batch, self.N), torch.ones(self.batch, self.N) * self.T_base], dim=1)
            y, I_traj = self.reservoir.solve(y, self.t_max, self.dt)
            pooled_traj = torch.nn.functional.max_pool1d(I_traj, kernel_size=self.len_y, stride=self.len_y)
            pooled_traj = pooled_traj.reshape(self.batch, self.N * self.len_t)
            out.append(pooled_traj)
        out = torch.cat(out, dim=0)
        out = out[:V_input.shape[0]]
        return out

    def forward(self, x):
        x = self.V_min + (self.V_max - self.V_min) * x
        reservoir_output = self.reservoir_func(x)
        reservoir_output = torch.cat([reservoir_output, torch.ones(reservoir_output.shape[0], 1)], dim=1)
        out = reservoir_output @ self.W
        return out

    def linear_regression(self, x, y):
        x = self.V_min + (self.V_max - self.V_min) * x
        reservoir_output = self.reservoir_func(x)
        reservoir_output = torch.cat([reservoir_output, torch.ones(reservoir_output.shape[0], 1)], dim=1)
        self.W = torch.linalg.lstsq(reservoir_output, y).solution
        prediction = reservoir_output @ self.W
        loss = torch.mean((prediction - y) ** 2)
        return loss


