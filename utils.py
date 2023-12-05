import warnings
import torch


def find_peaks(y, threshold, min_dist):
    # y: (batch, length)
    if min_dist % 2 == 0:
        warnings.warn('min_dist must be odd, but got {}'.format(min_dist))
        min_dist += 1
    batch = y.shape[0]
    local_maxima_mask = torch.cat([torch.zeros(batch, 1, dtype=torch.bool),
                                   (y[:, :-2] < y[:, 1:-1]) & (y[:, 2:] < y[:, 1:-1]),
                                   torch.zeros(batch, 1, dtype=torch.bool)], dim=1)
    threshold_mask = y > threshold

    _, indices = torch.nn.functional.max_pool1d(y, kernel_size=min_dist, stride=1,
                                                padding=min_dist // 2, return_indices=True)
    maxpool_mask = torch.zeros_like(y, dtype=torch.bool)
    maxpool_mask.scatter_(1, indices, True)

    peak_mask = local_maxima_mask & threshold_mask & maxpool_mask
    return peak_mask


def traj2peak(y, peak_threshold, min_dist):
    peak_mask = find_peaks(y, peak_threshold, min_dist)
    return peak_mask.sum(dim=0)


def bin_peaks(peaks, bin_size):
    return bin_size * torch.nn.functional.avg_pool1d(peaks.float().unsqueeze(0), bin_size, stride=bin_size).squeeze(0)


def bin_traj(peak_mask, len_x, len_y):
    return torch.nn.functional.avg_pool2d(peak_mask.float().unsqueeze(0), (len_x, len_y),
                                         stride=(len_x, len_y), divisor_override=1).squeeze(0)


