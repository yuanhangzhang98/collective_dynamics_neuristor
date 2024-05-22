import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
from scipy.stats import linregress
from scipy.special import zeta
from scipy.optimize import minimize

import plt_config
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # for running on server without GUI

color = plt.rcParams['axes.prop_cycle'].by_key()['color']

os.makedirs('fits', exist_ok=True)

folder = 'results'
Ns = [4096]
d = 2
V_batches = [np.array([9.96])]
R = 12
noise_strength = 0.001
Cth_factors = np.ones(1)
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

for V_batch in V_batches:
    xs_PDF = []
    ys_PDF = []
    xs_CDF = []
    ys_CDF = []
    slopes_PDF = []
    slopes_CDF = []
    intercepts_PDF = []
    intercepts_CDF = []
    alphas = []

    for N in Ns:
        for V in V_batch:
            variable_of_interest = 'Cth_factors'
            loop_var = variable_of_interest[:-1]

            for var in eval(variable_of_interest):
                exec(f'{loop_var} = {var}')
                name_string = f'{N}_{V:.3f}_{1000 * noise_strength:.4f}_{Cth_factor:.4f}_{couple_factor:.4f}_{width_factor:.4f}_' \
                              f'{len_x}_{len_y}'
                cluster_sizes = []
                n_files = int(N / 1024)
                for i in range(n_files):
                    try:
                        with open(f'avalanches/clusters_{name_string}_{i}.npy', 'rb') as f:
                            cluster_size_i = np.load(f)
                        cluster_sizes.append(cluster_size_i)
                    except FileNotFoundError:
                        continue
                cluster_sizes = np.concatenate(cluster_sizes)

                # # MLE estimation of exponent, assuming xmin = 1
                n_obs = len(cluster_sizes)
                # # continuous power law
                # alpha_continuous = 1 + n_obs / np.sum(np.log(cluster_sizes))
                #
                # # discrete power law
                # def neg_log_likelihood(alpha):
                #     return n_obs * np.log(zeta(alpha)) + alpha * np.sum(np.log(cluster_sizes))
                # try:
                #     res = minimize(neg_log_likelihood, alpha_continuous, bounds=[(1.01, None)])
                #     alpha_discrete = res.x[0]
                # except:
                #     alpha_discrete = np.nan
                # alphas.append(alpha_discrete)

                # tau = 2
                # c = N
                # M = int(c ** (tau - 1))
                # j = int((tau - 1) * N) ** ((tau - 1) / tau)
                # power_law_start = int((j / N) ** (1 - tau))
                # bin_edges = np.concatenate([np.arange(1, j+1),
                #                             np.floor(c * (np.arange(power_law_start, 0, -1)) ** (-1 / (tau - 1))).astype(int)])
                # bin_edges = bin_edges.astype(int)
                # bin_edges = np.unique(bin_edges)
                # bin_sizes = np.diff(bin_edges)
                # hist, bin_edges = np.histogram(cluster_sizes, bins=bin_edges)
                # bin_centers = bin_edges[:-1]  # approximation
                # hist = hist / bin_sizes
                # hist = hist / hist.sum()
                #
                # bin_centers = bin_centers[hist > 0]
                # hist = hist[hist > 0]

                log_cluster_sizes = np.log10(cluster_sizes)
                std = np.std(log_cluster_sizes)
                bin_width = 3.5 * std / (len(log_cluster_sizes) ** (1 / 3))
                bin_width = max(bin_width, 0.02)
                hist_min = 0
                hist_max = 6
                n_bins = int((hist_max - hist_min) / bin_width)
                n_bins = max(n_bins, 1)
                bins = bin_width * np.arange(n_bins + 1)
                bins = (10 ** bins).astype(int)
                bins = np.unique(bins)
                # bins = np.concatenate([
                #     np.arange(0, 10, 1),
                #     np.arange(10, 100, 2),
                #     np.arange(100, 1000, 20),
                #     np.arange(1000, 10000, 200),
                #     np.arange(10000, 100000, 2000),
                #     np.arange(100000, 1000000, 20000)
                # ])
                hist, bin_edges = np.histogram(cluster_sizes, bins=bins)
                bin_sizes = np.diff(bins)
                hist = hist / bin_sizes
                hist = hist / hist.sum()
                bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(int)
                bin_centers = bin_centers[hist > 0]
                hist = hist[hist > 0]
                bin_centers = np.log10(bin_centers)
                xs_PDF.append(bin_centers)
                ys_PDF.append(hist)

                try:
                    # fit = np.polyfit(np.log(bin_centers[fit_mask]), np.log(hist[fit_mask]), 1)
                    slope, intercept, r, p, se = linregress(bin_centers[:int(0.3*len(bin_centers))],
                                                            np.log10(hist[:int(0.3*len(bin_centers))]))
                    n_bins = len(bin_centers)
                    max_bin = bin_centers[-1]
                except:
                    slope, intercept, r, p, se = [np.nan, np.nan, np.nan, np.nan, np.nan]
                    n_bins = 0
                    max_bin = 0
                slopes_PDF.append(slope)
                intercepts_PDF.append(intercept)

                try:
                    fig, ax = plt.subplots()
                    ax.scatter(10 ** bin_centers, hist, s=10, alpha=0.5)
                    ax.plot(10 ** bin_centers, 10 ** (slope * bin_centers + intercept), '--', color='k')
                    ax.text(1, 10 ** -3, f'$\sim s^{{{slope:.2f}}}$', fontsize=18)
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.set_xlabel('Avalanche size s')
                    ax.set_ylabel('Probability P(s)')
                    plt.savefig(f'fits/power_law_PDF_{name_string}.png',
                                dpi=300, bbox_inches='tight')
                    plt.close()
                except ValueError:
                    plt.close()

                # cluster_sizes, cluster_counts = np.unique(cluster_sizes, return_counts=True)
                # CDF = np.cumsum(np.concatenate([[0], cluster_counts])) / np.sum(cluster_counts)
                # CCDF = 1 - CDF[:-1]
                # fit_mask = cluster_sizes < N ** 0.5
                # xs_CDF.append(cluster_sizes)
                # ys_CDF.append(CCDF)
                #
                # try:
                #     slope, intercept, r, p, se = linregress(np.log10(cluster_sizes[fit_mask]),
                #                                             np.log10(CCDF[fit_mask]))
                #     n_bins = len(cluster_sizes)
                #     max_bin = cluster_sizes[-1]
                # except:
                #     slope, intercept, r, p, se = [np.nan, np.nan, np.nan, np.nan, np.nan]
                #     n_bins = 0
                #     max_bin = 0
                # slopes_CDF.append(slope)
                # intercepts_CDF.append(intercept)
                # try:
                #     fig, ax = plt.subplots()
                #     ax.scatter(cluster_sizes, CCDF, s=10, alpha=0.5, label='Data')
                #     ax.plot(cluster_sizes, 10 ** (slope * np.log10(cluster_sizes) + intercept), '--', color='k',
                #             label=f'Fit $\sim s^{{{slope:.2f}}}$')
                #     ax.text(1, 10 ** -3, f'$\sim s^{{{slope:.2f}}}$', fontsize=18)
                #     ax.set_xscale('log')
                #     ax.set_yscale('log')
                #     ax.set_xlabel('Avalanche size s')
                #     ax.set_ylabel('CCDF')
                #     # ax.scatter(bin_centers, np.log10(hist), s=10, alpha=0.5)
                #     # ax.plot(bin_centers, slope * bin_centers + intercept, 'r--',
                #     #         label=f'{slope:.2f}x+{intercept:.2f} r={r:.2f}')
                #     # ax.set_yscale('log')
                #     # ax.set_xlabel('log10 (Avalanche Size)')
                #     # ax.set_ylabel('log10 (Probability)')
                #     plt.savefig(f'fits/power_law_CDF_{name_string}.png',
                #                 dpi=300, bbox_inches='tight')
                #     plt.close()
                #
                # except ValueError:
                #     plt.close()

                std = 0
                with open(f'fits/power_law_{name_string}.txt', 'w') as f:
                    f.write(f'{std}\t{slope}\t{intercept}\t'
                            f'{r}\t{p}\t{se}\t{n_bins}\t{max_bin}\t{n_obs}\n')
                print(f'{name_string}\t{std}\t{slope}\t{intercept}\t{r}\t{p}\t{se}\t{n_bins}\t{max_bin}\t'
                      f'{n_obs}')

    fig, ax = plt.subplots(figsize=(6, 4.5))
    # for i, N in enumerate(Ns):
    #     plt.scatter(10 ** xs_PDF[i], ys_PDF[i], s=10, alpha=0.5, label=f'N={N}  ~$s^{{{slopes_PDF[i]:.2f}}}$', color=color[i])
    #     plt.plot(10 ** xs_PDF[i], 10 ** (slopes_PDF[i] * xs_PDF[i] + intercepts_PDF[i]), '--', color=color[i])
    for i, N in enumerate(Ns):
        L = int(np.sqrt(N))
        plt.scatter(10 ** xs_PDF[i], ys_PDF[i], s=15, alpha=0.5, label=f'{L}x{L}', color=color[i])
    x_fit = 10 ** xs_PDF[0]
    y_fit = 10 ** (slopes_PDF[0] * xs_PDF[0] + intercepts_PDF[0])
    mask = y_fit > ys_PDF[0].min()
    plt.plot(x_fit[mask], y_fit[mask], '--', color='k')
    plt.text(5, 10 ** -1, f'$\sim s^{{{slopes_PDF[0]:.2f}}}$', fontsize=24)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Avalanche size s')
    plt.ylabel('Probability P(s)')
    # plt.legend()
    plt.savefig(f'fits/power_law_PDF_{V_batch[0]:.3f}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'fits/power_law_PDF_{V_batch[0]:.3f}.svg', dpi=300, bbox_inches='tight')
    plt.close()
