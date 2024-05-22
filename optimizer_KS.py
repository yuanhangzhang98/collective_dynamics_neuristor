import os
import numpy as np
import time
import torch
import torchvision
from tqdm import trange
from reservoir_KS import Reservoir2D
from KS_dataset import KSDataset


class Optimizer:
    def __init__(self, batch, V_min=11.0, V_max=13.0, Cth_factor=1.0, noise_strength=0.002):
        # self.batch = batch
        self.V_min = V_min
        self.V_max = V_max
        self.Cth_factor = Cth_factor
        self.noise_strength = noise_strength
        self.L = 16
        self.L_predict = 1
        self.l = 2
        self.n_steps = 1000
        self.dt = 0.05
        self.prediction_steps = 1
        self.train_set = KSDataset(5, self.n_steps, self.L, self.l, self.dt, self.prediction_steps)
        self.test_set = KSDataset(5, self.n_steps, self.L, self.l, self.dt, self.prediction_steps)
        self.Nx = self.L_predict + 2 * self.l
        self.Ny = self.L_predict + 2 * self.l
        self.N_out = self.L_predict * self.L_predict
        self.batch = batch
        self.reservoir = Reservoir2D(self.batch, self.Nx, self.Ny, self.N_out, self.V_min, self.V_max, self.Cth_factor,
                                     self.noise_strength)
        self.reservoir_test = Reservoir2D(self.L ** 2, self.Nx, self.Ny, self.N_out, self.V_min, self.V_max, self.Cth_factor,
                                     self.noise_strength)

    @staticmethod
    def normalize(u):
        return (u + 10) / 20

    def train(self):
        print('Performing linear regression......')
        t0 = time.time()
        # data = self.normalize(self.train_set.x).clamp(0, 1).reshape(-1, self.Nx * self.Ny)
        data = self.train_set.x.clamp(0, 1)
        target = self.train_set.y
        loss = self.reservoir.linear_regression(data, target).item()
        t1 = time.time()
        print(f'Loss = {loss:.4f}, Time = {t1 - t0:.2f}s')

    def test(self):
        # data = self.normalize(self.test_set.x).clamp(0, 1).reshape(-1, self.Nx * self.Ny)
        data = self.test_set.x.clamp(0, 1)
        target = self.test_set.y
        prediction = self.reservoir.forward(data)
        test_loss = torch.mean((prediction - target) ** 2).item()
        print(f'Test set: Average loss: {test_loss:.4f}')
        return test_loss

    @torch.no_grad()
    def predict_dynamics(self, u0, steps=100):
        self.reservoir_test.W = self.reservoir.W
        u = u0
        trajectory = [u.clone()]
        print('Predicting with NN dynamics......')
        pad_op = torch.nn.CircularPad2d(self.l)
        for t in trange(steps):
            x = pad_op(self.normalize(u - u.mean()).clamp(0, 1).reshape(1, self.L, self.L))
            kernel = 2 * self.l + 1
            x = x.unfold(1, kernel, 1).unfold(2, kernel, 1).reshape(self.L ** 2, kernel ** 2)
            du = self.reservoir_test.forward(x).reshape(u0.shape)
            u = u + du
            trajectory.append(u.clone())
        trajectory = torch.stack(trajectory, dim=0)
        return trajectory

    def hyperopt_objective(self, param):
        reservoir = Reservoir2D(self.batch, self.Nx, self.Ny, self.N_out, param['V_min'], param['V_max'],
                                param['Cth'], self.noise_strength)
        data = self.train_set.x.clamp(0, 1).reshape(-1, self.Nx * self.Ny)
        target = self.train_set.y.reshape(-1, self.N_out)
        loss = reservoir.linear_regression(data, target).item()
        test_data = self.test_set.x.clamp(0, 1).reshape(-1, self.Nx * self.Ny)
        test_target = self.test_set.y.reshape(-1, self.N_out)
        prediction = reservoir.forward(test_data)
        test_loss = torch.mean((prediction - test_target) ** 2).item()
        return test_loss


if __name__ == '__main__':
    torch.set_default_tensor_type(torch.cuda.FloatTensor
                                  if torch.cuda.is_available()
                                  else torch.FloatTensor)
    try:
        os.mkdir('ckpts')
    except FileExistsError:
        pass
    try:
        os.mkdir('results')
    except FileExistsError:
        pass
    batch = int(5e4)
    V_min = 12.413381282561543
    V_max = 12.19013679073254
    Cth_factor = 0.43684498014118855
    noise_strength = 0
    optimizer = Optimizer(batch, V_min, V_max, Cth_factor, noise_strength)
    test_traj = optimizer.test_set.trajectory[:, 0]
    torch.save(test_traj, f'ckpts/test_traj.pt')
    optimizer.train()
    test_loss = optimizer.test()
    torch.save(optimizer.reservoir.W, f'ckpts/reservoir.pt')
    with open('results/log.txt', 'a') as f:
        f.write(f'{V_min} {V_max} {Cth_factor} {noise_strength} {test_loss}\n')

    # Predict dynamics
    predicted_traj = optimizer.predict_dynamics(test_traj[0], 100)
    torch.save(predicted_traj, f'ckpts/predicted_traj.pt')
