import numpy as np
import torch
import time
pi = np.pi


def P(x, gamma):
    return 0.5 * (1 - torch.sin(gamma * x)) * (1 + torch.tanh(pi ** 2 - 2 * pi * x))


class VO2:
    def __init__(self, N, width_factor=1):
        self.N = N
        self.width_factor = width_factor
        self.w = 7.19357064e+00 * width_factor
        self.Tc = 3.32805839e+02
        self.beta = 2.52796285e-01
        self.R0 = 5.35882879e-03
        self.Ea = 5.22047417e+03
        self.gamma = 9.56269682e-01
        self.Rm0 = 262.5
        self.Rm_factor = 4.90025335
        self.Rm = self.Rm0 * self.Rm_factor
        self.delta = torch.ones(N)
        self.reversed = torch.zeros(N)
        self.Tr = None
        self.gr = None
        self.Tpr = None
        self.T_last = None

    def initialize(self, T0):
        T = T0 * torch.ones(self.N)
        self.gr = self.g_major(T)
        self.Tr = T
        self.Tpr = self.Tpr_func()
        self.T_last = T

    def reversal(self, T):
        T = T.clamp(305, 370)
        dT = T - self.T_last
        if dT.abs().max() > 0.01:
            delta = torch.sign(dT)
            reversal_mask = (delta != self.delta) & (delta != 0)
            if reversal_mask.any():
                self.gr[reversal_mask] = self.g(T)[reversal_mask]
                self.delta[reversal_mask] = delta[reversal_mask]
                self.reversed[reversal_mask] = 1
                self.Tr[reversal_mask] = T[reversal_mask]
                self.Tpr[reversal_mask] = self.Tpr_func()[reversal_mask]
                # print('Reversal temperature: {:.4f} K'.format(T))
            self.T_last = T

    def Tpr_func(self):
        return self.delta * self.w / 2 + self.Tc - torch.arctanh(2 * self.gr - 1) / self.beta - self.Tr

    def g_major(self, T):
        return 0.5 + 0.5 * torch.tanh(self.beta * (self.delta * self.w / 2 + self.Tc - T))

    def g_minor(self, T):
        return 0.5 + 0.5 * torch.tanh(self.beta * (self.delta * self.w / 2 + self.Tc -
                                                (T + self.Tpr * P((T - self.Tr) / self.Tpr, self.gamma))))

    def g(self, T):
        Tp = self.Tpr * P((T - self.Tr) / (self.Tpr + 1e-6), self.gamma) * self.reversed
        return 0.5 + 0.5 * torch.tanh(self.beta * (self.delta * self.w / 2 + self.Tc - (T + Tp)))

    def R(self, T):
        # unit: kOhm
        T = T.clamp(305, 370)
        return (self.R0 * torch.exp(self.Ea / T) * self.g(T) + self.Rm) / 1000


# param_optim = np.array([145.34619293,  49.62776831,   0.20558726,   4.90025335])
# C, Cth, Sth, Rm_factor = params
class Circuit:
    def __init__(self, batch, N, V, R, noise_strength, Cth_factor, couple_factor, width_factor, T_base=325):
        self.batch = batch
        self.N = N
        self.d = 1
        self.V0 = V * torch.ones(self.batch, self.N)  # V
        self.R0 = R  # kOhm
        self.C0 = 145.34619293  # pF
        self.R0C0 = self.R0 * self.C0
        self.Cth_factor = Cth_factor
        self.Cth = 49.62776831  # mW * ns / K
        self.Sth = 0.20558726  # mW / K
        self.couple_factor = couple_factor
        self.S_env = self.Sth * (1 - 2 * self.d * self.couple_factor)
        self.S_couple = self.couple_factor * self.Sth
        self.noise_strength = noise_strength
        self.width_factor = width_factor
        self.T_base = T_base

        self.VO2 = VO2(batch*N, width_factor)
        self.VO2.initialize(self.T_base - 0.1)

        self.IR = None
        self.T = None
        self.R = None

        self.compiled_step = None

    def set_input(self, V=None, Cth_factor=None):
        if V is not None:
            self.V0 = V  # (batch, N)
        if Cth_factor is not None:
            self.Cth_factor = Cth_factor  # (batch, N)

    def dydt(self, t, y):
        V1 = y[:, 0, :]  # (batch, N)
        T = y[:, 1, :]  # (batch, N)
        T_padded = torch.cat([T[:, 0], T, T[:, -1]], dim=0)
        laplacian = T_padded[:, -2] - 2 * T + T_padded[:, 2:]

        R = self.VO2.R(T.reshape(-1)).reshape(self.batch, self.N)
        IR = V1 / R
        self.IR = IR
        # self.T = T
        # self.R = R
        QR = IR ** 2 * R
        dV1 = self.V0 / self.R0C0 - V1 / self.R0C0 - V1 / (R * self.C0)
        dT = ((QR - self.S_env * (T - self.T_base) + self.S_couple * laplacian) / self.Cth
             + self.noise_strength * torch.randn_like(T)) / self.Cth_factor
        return torch.stack([dV1, dT], dim=1)

    def step(self, t, y):
        self.VO2.reversal(y[:, 1].reshape(-1))
        dy = self.dydt(t, y)
        return dy

    @torch.no_grad()
    def solve(self, y0, t_max, dt):
        t = 0
        # dt = 10
        y = y0
        t_traj = []
        I_traj = []
        # R_traj = []
        # T_traj = []
        n_max = int(t_max / dt)

        if self.compiled_step is None:
            self.compiled_step = torch.compile(self.step)
        # self.compiled_step = self.step

        t0 = time.time()
        for i in range(n_max):
            dy = self.compiled_step(t, y)
            # dy = self.step(t, y)
            t += dt
            y += dy * dt
            t_traj.append(t)
            I_traj.append(self.IR.detach().clone())
            # R_traj.append(self.R.detach().clone())
            # T_traj.append(self.T.detach().clone())

            # if i % 1000 == 0:
            #     print(f'Noise={1000 * self.noise_strength:.4f}  '
            #           f'Couple={self.couple_factor:.4f}  Width={self.width_factor:.4f}  '
            #           f'Step: {int(i/1000)} / {int(n_max/1000)}  Time: {time.time() - t0:.2f} s')
            #     t0 = time.time()

        return y, torch.stack(I_traj, dim=-1)

class Circuit2D(Circuit):
    def __init__(self, batch, Nx, Ny, V, R, noise_strength, Cth_factor, couple_factor, width_factor, T_base=325):
        N = Nx * Ny
        super().__init__(batch, N, V, R, noise_strength, Cth_factor, couple_factor, width_factor, T_base)
        self.Nx = Nx
        self.Ny = Ny
        self.d = 2
        self.S_env = self.Sth * (1 - 2 * self.d * self.couple_factor)

    def dydt(self, t, y):
        V1 = y[:, 0, :]  # (batch, N)
        T = y[:, 1, :]  # (batch, N)
        T_2D = T.view(self.batch, self.Nx, self.Ny)
        T_padded = torch.nn.functional.pad(T_2D, (1, 1, 1, 1), mode='replicate')  # (batch, Nx+2, Ny+2)
        laplacian = T_padded[:, :-2, 1:-1] + T_padded[:, 2:, 1:-1] + T_padded[:, 1:-1, :-2] + T_padded[:, 1:-1, 2:] \
                    - 4 * T_2D
        laplacian = laplacian.view(self.batch, self.N)  # (N, )

        R = self.VO2.R(T.reshape(-1)).reshape(self.batch, self.N)
        IR = V1 / R
        self.IR = IR
        # self.T = T
        # self.R = R
        QR = IR ** 2 * R
        dV1 = self.V0 / self.R0C0 - V1 / self.R0C0 - V1 / (R * self.C0)
        dT = ((QR - self.S_env * (T - self.T_base) + self.S_couple * laplacian) / self.Cth
              + self.noise_strength * torch.randn_like(T)) / self.Cth_factor
        return torch.stack([dV1, dT], dim=1)

if __name__ == '__main__':
    # Test the hysteresis model
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    VO2_model = VO2(1)
    T = torch.tensor([325, 360, 325, 360, 330, 360, 335, 360], dtype=torch.float)
    T_last = 324.9
    VO2_model.initialize(T_last)
    Rs = []
    Ts = []
    for i, Ti in enumerate(T[1:]):
        T_array = torch.linspace(T_last, Ti, 100)
        for Tii in T_array:
            Tii = Tii.unsqueeze(0)
            VO2_model.reversal(Tii)
            Ri = VO2_model.R(Tii)
            Rs.append(Ri)
        T_last = Ti
        Ts.append(T_array)
    Rs = torch.cat(Rs, dim=0).detach().cpu().numpy()
    Ts = torch.cat(Ts, dim=0).detach().cpu().numpy()

    import matplotlib.pyplot as plt
    import plt_config
    plt.plot(Ts, Rs)
    plt.yscale('log')
    plt.xlim([320, 360])
    plt.ylim([1, 60])
    plt.xlabel('Temperature (K)')
    plt.ylabel('Resistance (kOhm)')
    plt.savefig('results/VO2_R.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()