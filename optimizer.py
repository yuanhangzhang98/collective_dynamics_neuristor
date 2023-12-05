import os
import numpy as np
import time
import torch
import torchvision
from reservoir import Reservoir2D


class Optimizer:
    def __init__(self, batch, V_min=11.0, V_max=13.0, Cth_factor=1.0, noise_strength=0.002):
        self.batch = batch
        self.V_min = V_min
        self.V_max = V_max
        self.Cth_factor = Cth_factor
        self.noise_strength = noise_strength
        transform = torchvision.transforms.ToTensor()
        self.train_set = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        self.test_set = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch, shuffle=True,
                                                        generator=torch.Generator(device='cuda'))
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch, shuffle=True,
                                                       generator=torch.Generator(device='cuda'))
        self.Nx = 28
        self.Ny = 28
        self.N_out = 10
        self.reservoir = Reservoir2D(self.batch, self.Nx, self.Ny, self.N_out, self.V_min, self.V_max, self.Cth_factor,
                                     self.noise_strength)
        self.compiled_model = torch.compile(self.reservoir)
        self.optimizer = torch.optim.Adam(self.reservoir.parameters(), lr=0.001)
        self.loss_func = torch.nn.NLLLoss()

    def train(self, epoch):
        for batch_idx, (data, target) in enumerate(self.train_loader):
            t0 = time.time()
            data = data.to('cuda')
            target = target.to('cuda')
            data = data.reshape(self.batch, self.Nx * self.Ny)
            self.optimizer.zero_grad()
            # output = self.reservoir(data)
            output = self.compiled_model(data)
            loss = self.loss_func(output, target)
            loss.backward()
            self.optimizer.step()
            # if batch_idx % 100 == 0:
            with open('results/train_loss.txt', 'a') as f:
                f.write(f'{batch_idx}\t{loss.item()}\n')
            t1 = time.time()
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                  f'({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tt: {t1 - t0:.4f}s')

    def test(self):
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to('cuda')
                target = target.to('cuda')
                data = data.reshape(self.batch, self.Nx * self.Ny)
                output = self.compiled_model(data)
                test_loss += self.loss_func(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss *= (self.batch / len(self.test_loader.dataset))
        accuracy = correct / len(self.test_loader.dataset)
        with open('results/test_loss.txt', 'a') as f:
            f.write(f'{test_loss}\n')
        with open('results/test_accuracy.txt', 'a') as f:
            f.write(f'{accuracy}\n')
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.test_loader.dataset)} '
              f'({accuracy:.4f})\n')
        return accuracy

    def param_optim_step(self, V_min, V_max, Cth_factor, noise_strength):
        self.reservoir.reset(V_min, V_max, Cth_factor, noise_strength)
        self.optimizer = torch.optim.Adam(self.reservoir.parameters(), lr=0.001)
        self.train(1)
        accuracy = self.test()
        with open('results/log.txt', 'a') as f:
            f.write(f'{V_min} {V_max} {Cth_factor} {noise_strength} {accuracy}\n')
        return -accuracy



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
    batch = 1000
    V_min = 10.5
    V_max = 12.2
    Cth_factor = 0.15
    noise_strength = 2e-4
    optimizer = Optimizer(batch, V_min, V_max, Cth_factor, noise_strength)
    for epoch in range(1, 21):
        optimizer.train(epoch)
        accuracy = optimizer.test()
        with open('results/log.txt', 'a') as f:
            f.write(f'{V_min} {V_max} {Cth_factor} {noise_strength} {accuracy}\n')