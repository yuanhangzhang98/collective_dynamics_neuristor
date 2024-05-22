import numpy as np
import torch
from tqdm import trange


def KS(u):
    pad_op = torch.nn.CircularPad2d(2)
    u = pad_op(u)
    ux = (u[:, 3:-1, 2:-2] - u[:, 1:-3, 2:-2]) / 2
    uy = (u[:, 2:-2, 3:-1] - u[:, 2:-2, 1:-3]) / 2
    ux2 = (ux ** 2 + uy ** 2)
    uxx = u[:, 3:-1, 2:-2] + u[:, 1:-3, 2:-2] + u[:, 2:-2, 3:-1] + u[:, 2:-2, 1:-3] - 4 * u[:, 2:-2, 2:-2]
    uxxxx = u[:, 4:, 2:-2] - 4 * u[:, 3:-1, 2:-2] + 12 * u[:, 2:-2, 2:-2] - 4 * u[:, 1:-3, 2:-2] + u[:, :-4, 2:-2] \
          + u[:, 2:-2, 4:] - 4 * u[:, 2:-2, 3:-1]                         - 4 * u[:, 2:-2, 1:-3] + u[:, 2:-2, :-4]
    # ut = -uxx - uxxxx - ux2 / 2 + ux2.mean() / 2
    ut = -uxx - uxxxx - ux2 / 2
    return ut


def RK4(u, dt):
    ut0 = KS(u)
    ut1 = KS(u + ut0 * dt / 2)
    ut2 = KS(u + ut1 * dt / 2)
    ut3 = KS(u + ut2 * dt)
    u = u + (ut0 + 2 * ut1 + 2 * ut2 + ut3) * dt / 6
    return u


def dynamics(u0, dt, n_steps):
    u = u0
    trajectory = []
    print('Simulating KS equation......')
    for t in trange(n_steps):
        u = RK4(u, dt)
        trajectory.append(u.clone())
    trajectory = torch.stack(trajectory, dim=0)
    return trajectory


if __name__ == '__main__':
    import os
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')
    os.makedirs('images', exist_ok=True)

    # Initial condition
    u0 = torch.randn(100, 16, 16)
    u0 = u0 - u0.mean()

    # Time step and number of steps
    dt = 0.01
    n_steps = 10000
    prediction_steps = 100

    # Dynamics
    trajectory = dynamics(u0, dt, n_steps)

    for i in range(n_steps // prediction_steps):
        plt.imshow(trajectory[100*i].numpy(), cmap='jet', vmin=-10, vmax=10)
        # plt.imshow(trajectory[prediction_steps * i].numpy(), cmap='jet')
        plt.colorbar()
        plt.axis('off')
        plt.savefig(f'images/{i:04d}.png')
        plt.close()
        print(f'Step {i+1}/{n_steps // prediction_steps}')
