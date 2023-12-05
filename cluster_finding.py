import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.setrecursionlimit(3000)


def find(vertex, parent):
    if parent[vertex] != vertex:
        parent[vertex] = find(parent[vertex], parent)
    return parent[vertex]


def union(u, v, parent):
    root_u = find(u, parent)
    root_v = find(v, parent)
    if root_u != root_v:
        parent[root_u] = root_v

def find_cluster(lattice):
    h, w = lattice.shape
    x = lattice > 0
    label = torch.zeros_like(x, dtype=torch.int)
    row_last = torch.zeros(w, dtype=bool)
    label_last = torch.zeros(w, dtype=torch.int)
    current_label = 0
    equivalence = []
    for i in range(h):
        row = x[i]
        leftmost_mask = torch.cat([row[0:1], row[1:] > row[:-1]], dim=0)
        label_i = (torch.cumsum(leftmost_mask, dim=0) + current_label) * row
        current_label += leftmost_mask.sum()

        label[i] = label_i.clone()

        label_last_padded = torch.nn.functional.pad(label_last, (1, 1))
        row_last_padded = torch.nn.functional.pad(row_last, (1, 1))
        equivalence_i = torch.cat([
            torch.stack([label_i, label_last_padded[1:-1]], dim=1)[row & row_last_padded[1:-1]],
            torch.stack([label_i, label_last_padded[2:]], dim=1)[row & row_last_padded[2:]],
            torch.stack([label_i, label_last_padded[:-2]], dim=1)[row & row_last_padded[:-2]]], dim=0)
        equivalence_i = torch.unique(equivalence_i, dim=0)
        equivalence.append(equivalence_i)

        row_last = row.clone()
        label_last = label_i.clone()

    equivalence = torch.cat(equivalence, dim=0)

    # find connected components of the equivalence graph
    edges = equivalence.cpu().numpy()
    nodes = np.arange(1, current_label.cpu().numpy().item() + 1)
    parent = {key: value for key, value in zip(nodes, nodes)}

    for edge in edges:
        union(edge[0], edge[1], parent)

    value_map = torch.tensor([find(node, parent) for node in nodes])
    unique_labels = torch.unique(value_map)
    if unique_labels.numel() == 0:
        return label, torch.zeros(1)
    else:
        relabeled = torch.arange(1, len(unique_labels) + 1, dtype=torch.int)
        relabel_map = torch.zeros(unique_labels.max() + 1, dtype=torch.int)
        relabel_map[unique_labels] = relabeled
        value_map = relabel_map[value_map]

        value_map = torch.cat([torch.zeros(1, dtype=torch.int), value_map], dim=0)

        # relabel the lattice
        label = value_map[label]

        index = label.reshape(-1).to(torch.int64)
        weight = lattice.reshape(-1)
        cluster_sizes = torch.zeros(value_map.max() + 1, dtype=torch.float)
        cluster_sizes.scatter_add_(0, index, weight)
        cluster_sizes = cluster_sizes[1:]

        return label, cluster_sizes


def find_cluster_2d(lattice, Nx, Ny):
    N, length = lattice.shape
    assert Nx * Ny == N, "Nx * Ny must equal to N"
    x = lattice > 0

    # Extend each peak forward by 1 so that I don't need to take care of the diagonals
    # As long as the window size is smaller than ISI/3, peaks will not overlap
    x = x | torch.cat([torch.zeros(N, 1, dtype=torch.bool), x[:, :-1]], dim=1)

    x = x.reshape(Nx, Ny, length)

    label = torch.zeros(Nx, Ny, length, dtype=torch.int)

    current_label = 0

    def label_row(row):
        nonlocal current_label
        leftmost_mask = torch.cat([row[0:1], row[1:] > row[:-1]], dim=0)
        label_i = (torch.cumsum(leftmost_mask, dim=0) + current_label) * row
        current_label += leftmost_mask.sum()
        return label_i

    for i in range(Nx):
        for j in range(Ny):
            label[i, j, :] = label_row(x[i, j, :])
    label_padded = torch.nn.functional.pad(label.T, (1, 1, 1, 1), mode='constant', value=0).T

    equivalence = torch.cat([
        torch.stack([label, label_padded[1:-1, 2:]], dim=3),
        torch.stack([label, label_padded[1:-1, :-2]], dim=3),
        torch.stack([label, label_padded[2:, 1:-1]], dim=3),
        torch.stack([label, label_padded[:-2, 1:-1]], dim=3)], dim=2)  # (Nx, Ny, 4*length, 2)
    equivalence = equivalence.reshape(Nx * Ny * 4 * length, 2)
    nonzero_mask = (equivalence > 0).all(dim=1)
    equivalence = equivalence[nonzero_mask]
    equivalence = torch.unique(equivalence, dim=0)

    # find connected components of the equivalence graph
    edges = equivalence.cpu().numpy()
    nodes = np.arange(1, current_label.cpu().numpy().item() + 1)
    parent = {key: value for key, value in zip(nodes, nodes)}

    for edge in edges:
        union(edge[0], edge[1], parent)

    label = label.reshape(N, length)
    value_map = torch.tensor([find(node, parent) for node in nodes])
    unique_labels = torch.unique(value_map)
    if unique_labels.numel() == 0:
        return label, torch.zeros(1)
    else:
        relabeled = torch.arange(1, len(unique_labels) + 1, dtype=torch.int)
        relabel_map = torch.zeros(unique_labels.max() + 1, dtype=torch.int)
        relabel_map[unique_labels] = relabeled
        value_map = relabel_map[value_map]

        value_map = torch.cat([torch.zeros(1, dtype=torch.int), value_map], dim=0)

        # relabel the lattice
        label = value_map[label]

        index = label.reshape(-1).to(torch.int64)
        weight = lattice.reshape(-1)
        cluster_sizes = torch.zeros(value_map.max() + 1, dtype=torch.float)
        cluster_sizes.scatter_add_(0, index, weight)
        cluster_sizes = cluster_sizes[1:]

        return label, cluster_sizes  # no need to divide by 2, already accounted for by weight

if __name__ == '__main__':
    # Create a sample 2D grid with random 0s and 1s
    grid = torch.randint(0, 2, (8, 8), dtype=torch.int)
    grid[grid > 0] = torch.randint(1, 10, (grid[grid > 0].shape), dtype=torch.int)
    print("Original Grid:")
    print(grid)
    print(f'Original sum: {grid.sum()}')

    # Apply parallel Hoshen-Kopelman algorithm
    labeled_grid, cluster_sizes = find_cluster(grid.clone())
    print("Labeled Grid:")
    print(labeled_grid)
    print("Cluster Sizes:")
    print(cluster_sizes)