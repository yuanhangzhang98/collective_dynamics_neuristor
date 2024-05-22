import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use('Agg')
os.makedirs('images', exist_ok=True)

test_traj = torch.load('ckpts/test_traj.pt')
test_traj = test_traj - test_traj.mean(dim=(1, 2), keepdim=True)
test_traj = (test_traj + 10) / 20  # [-10, 10] -> [0, 1]
prediction_steps = 1
test_traj = test_traj[::prediction_steps]

predicted_traj = torch.load(f'ckpts/predicted_traj.pt')
predicted_traj = predicted_traj - predicted_traj.mean(dim=(1, 2), keepdim=True)
predicted_traj = (predicted_traj + 10) / 20  # [-10, 10] -> [0, 1]

n_steps = min(len(predicted_traj), len(test_traj))
test_traj = test_traj[:n_steps]
predicted_traj = predicted_traj[:n_steps]

difference = predicted_traj - test_traj

for i in range(n_steps):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    im1 = ax1.imshow(test_traj[i].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    ax1.set_title('Test')
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(predicted_traj[i].cpu().numpy(), cmap='jet', vmin=0, vmax=1)
    ax2.set_title('Predicted')
    plt.colorbar(im2, ax=ax2)

    im3 = ax3.imshow(difference[i].cpu().numpy(), cmap='jet', vmin=-0.5, vmax=0.5)
    ax3.set_title('Difference')
    plt.colorbar(im3, ax=ax3)

    plt.tight_layout()
    plt.savefig(f'images/{i:03d}.png')
    plt.close()
    print('Saved image', i)