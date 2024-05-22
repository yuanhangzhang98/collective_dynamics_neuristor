import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from KS_equation import dynamics


class KSDataset(Dataset):
    def __init__(self, batch, n_steps, L, l, dt, prediction_steps):
        super().__init__()
        self.batch = batch
        self.n_steps = n_steps
        self.L = L
        self.l = l
        self.dt = dt
        self.prediction_steps = prediction_steps
        self.x = None
        self.y = None
        self.trajectory = None
        self.reset()

    def reset(self):
        u0 = torch.randn(self.batch, self.L, self.L)
        u0 = u0 - u0.mean(dim=(1, 2), keepdim=True)
        trajectory = dynamics(u0, self.dt, self.n_steps)
        trajectory = trajectory[int(0.2*self.n_steps):]
        self.trajectory = trajectory
        trajectory_centered = trajectory - trajectory.mean(dim=(2, 3), keepdim=True)
        trajectory_normalized = (trajectory_centered + 10) / 20  # [-10, 10] -> [0, 1]  # (length, batch, L, L)
        kernel = 2 * self.l + 1
        pad_op = torch.nn.CircularPad2d(self.l)
        traj_padded = pad_op(trajectory_normalized[:-self.prediction_steps])
        length = traj_padded.shape[0]
        self.x = traj_padded.unfold(2, kernel, 1).unfold(3, kernel, 1)\
            .reshape(length * self.batch * self.L ** 2, kernel ** 2)
        self.y = (self.trajectory[self.prediction_steps:] - self.trajectory[:-self.prediction_steps])\
            .reshape(length * self.batch * self.L ** 2, 1)
        # self.trajectory = trajectory
        # self.x = trajectory[:-self.prediction_steps]
        # self.y = trajectory[self.prediction_steps:]
        # self.y = trajectory[self.prediction_steps:] - trajectory[:-self.prediction_steps]
        # self.y = self.trajectory[self.prediction_steps:]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
