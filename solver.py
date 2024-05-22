import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
from model import Circuit, Circuit2D
from utils import find_peaks, bin_traj
from cluster_finding import find_cluster, find_cluster_2d
from tqdm import trange

import matplotlib
import plt_config
import matplotlib.pyplot as plt
matplotlib.use('Agg') # for running on server without GUI

color = plt.rcParams['axes.prop_cycle'].by_key()['color']

torch.set_default_tensor_type(torch.cuda.FloatTensor
                                    if torch.cuda.is_available()
                                    else torch.FloatTensor)


class Solver:
    def __init__(self, d, batch, N, V, R, noise_strength, Cth_factor, couple_factor, width_factor, T_base,
                 t_max, dt, n_repeat, peak_threshold, min_dist, len_x=2, len_y=50, batch_idx=0):
        self.d = d
        if d == 1:
            self.N = N
        elif d == 2:
            self.Nx = int(np.sqrt(N))
            self.Ny = int(np.sqrt(N))
            self.N = self.Nx * self.Ny
        else:
            raise NotImplementedError(f'Only 1D and 2D are supported, got {d}')
        self.batch = batch
        self.N = N
        self.V = V
        self.R = R
        self.noise_strength = noise_strength
        self.Cth_factor = Cth_factor
        self.couple_factor = couple_factor
        self.width_factor = width_factor
        self.T_base = T_base
        self.t_max = t_max
        self.dt = dt
        self.n_repeat = n_repeat
        self.peak_threshold = peak_threshold
        self.min_dist = min_dist
        self.save_length = 1000
        self.transient = 0.5
        self.len_x = len_x
        self.len_y = len_y
        self.batch_idx = batch_idx

        self.name_string = f'{N}_{V:.3f}_{1000 * noise_strength:.4f}_{Cth_factor:.4f}_{couple_factor:.4f}_' \
                           f'{width_factor:.4f}'

        if d == 1:
            self.model = Circuit(batch, N, V, R, noise_strength, Cth_factor, couple_factor, width_factor, T_base)
        elif d == 2:
            self.model = Circuit2D(batch, self.Nx, self.Ny, V, R, noise_strength, Cth_factor, couple_factor, width_factor,
                                   T_base)
        else:
            raise NotImplementedError(f'Only 1D and 2D are supported, got {d}')

    def dynamics(self, save_traj=False, plot_2D=False):
        # peaks = []
        peaks = torch.zeros(self.batch, int(self.N / self.len_x), self.n_repeat,
                            int(self.t_max / self.n_repeat / self.dt / self.len_y))
        y = torch.stack([torch.zeros(self.batch, self.N), torch.ones(self.batch, self.N) * self.T_base], dim=1)
        t_max_i = int(self.t_max / self.n_repeat)
        plot_epochs = 1
        print('Simulating dynamics...')
        for i in trange(1, self.n_repeat + 1):
            y, I_traj = self.model.solve(y, t_max_i, self.dt)
            peak_i = find_peaks(I_traj.reshape(self.batch * self.N, -1), self.peak_threshold, self.min_dist)
            peak_i = peak_i.reshape(self.batch, self.N, -1)
            peak_i = bin_traj(peak_i, self.len_x, self.len_y)
            peaks[:, :, i-1] = peak_i
            if save_traj:
                torch.save(I_traj, f'results/I_traj_{self.name_string}_{i}.pt')
            if self.d == 2 and plot_2D and i > self.n_repeat - plot_epochs:
                for batch_idx in trange(1):
                    V = self.model.V0[batch_idx].item()
                    Cth_factor = self.model.Cth_factor[batch_idx].item()
                    name_string_i = f'{self.N}_{V:.3f}_{1000 * self.noise_strength:.4f}_{Cth_factor:.4f}_' \
                                    f'{self.couple_factor:.4f}_{self.width_factor:.4f}'
                    I_traj_i = I_traj[batch_idx].reshape(self.Nx, self.Ny, -1).cpu().numpy()
                    n_step = I_traj.shape[2]

                    step_interval = 16
                    plot_idx = list(range(0, n_step, step_interval))
                    for j in plot_idx:
                        fig, ax = plt.subplots()
                        ax.imshow(I_traj_i[:, :, j], cmap='Blues', norm=matplotlib.colors.Normalize(vmin=0, vmax=I_traj.max()))
                        ax.set_axis_off()
                        fig.savefig(f'graphs/2D_{name_string_i}_{i}_{j}.png', dpi=300, bbox_inches='tight')
                        fig.savefig(f'graphs/2D_{name_string_i}_{i}_{j}.svg', dpi=300, bbox_inches='tight')
                        plt.close()
            del I_traj

        print(f'{self.name_string} done')
        peaks = peaks.reshape(self.batch, self.N, -1)
        return peaks

    def find_avalanches(self, binned_traj, save_binned_peaks=False, ensemble=False):
        binned_traj = binned_traj[:, :, int(binned_traj.shape[2] * self.transient):]
        if save_binned_peaks:
            torch.save(binned_traj, f'peaks/binned_peaks_{self.name_string}_{self.len_x}_{self.len_y}.pt')
        print('Finding avalanches......')
        if ensemble:
            cluster_sizes_batch = []
        for i in trange(self.batch):
            binned_traj_i = binned_traj[i]
            if self.d == 1:
                _, cluster_sizes = find_cluster(binned_traj_i)
            elif self.d == 2:
                _, cluster_sizes = find_cluster_2d(binned_traj_i, self.Nx, self.Ny)
            if ensemble:
                cluster_sizes_batch.append(cluster_sizes)
                if i == self.batch - 1:
                    cluster_sizes = torch.cat(cluster_sizes_batch, dim=0)
                else:
                    continue

            cluster_sizes = cluster_sizes[cluster_sizes > 0].cpu().numpy()
            if len(cluster_sizes) < 2:
                continue
            V = self.model.V0[i].item()
            Cth_factor = self.model.Cth_factor[i].item()
            name_string_i = f'{self.N}_{V:.3f}_{1000 * self.noise_strength:.4f}_{Cth_factor:.4f}_' \
                            f'{self.couple_factor:.4f}_{self.width_factor:.4f}'
            log_cluster_sizes = np.log10(cluster_sizes)
            std = np.std(log_cluster_sizes)
            bin_width = 3.5 * std / (len(log_cluster_sizes) ** (1 / 3))
            bin_width = max(bin_width, 0.02)
            hist_min = 0
            hist_max = 6
            n_bins = int((hist_max - hist_min) / bin_width)
            n_bins = max(n_bins, 1)
            bins = bin_width * np.arange(n_bins + 1)
            hist, bin_edges = np.histogram(log_cluster_sizes, bins=bins)
            bin_edges_linear = 10 ** bin_edges
            bin_sizes_linear = np.diff(bin_edges_linear)
            hist = hist / bin_sizes_linear
            hist = hist / hist.sum()
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_centers = bin_centers[hist > 0]
            hist = hist[hist > 0]
            # slope, intercept, r, p, se = linregress(bin_centers, np.log10(hist))
            with open(f'avalanches/clusters_{name_string_i}_{self.len_x}_{self.len_y}_{self.batch_idx}.npy', 'wb') as f:
                np.save(f, cluster_sizes)

        return cluster_sizes
    def run(self, save_traj=False, save_binned_peaks=False, plot_2D=False, ensemble=False):
        binned_peaks = self.dynamics(save_traj, plot_2D)
        cluster_sizes = self.find_avalanches(binned_peaks, save_binned_peaks, ensemble)
        return cluster_sizes
