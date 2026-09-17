import torch
from kan import KAN

import config as cfg


def build_kan_width(n_species):
    return [n_species, 16, 16, 16, n_species]


def build_species_weight(species_names, device):
    return torch.tensor(
        [cfg.SPECIES_WEIGHTS.get(name, 1.0) for name in species_names],
        dtype=torch.float32, device=device,
    )


def make_weighted_mse(species_weight):
    def weighted_mse(pred, target):
        return torch.mean(species_weight * (pred - target) ** 2)
    return weighted_mse


def make_model(kan_width, seed, device):
    torch.manual_seed(seed)
    m = KAN(width=kan_width, grid=5, k=3, seed=seed, device=device, auto_save=False, grid_eps=0.0)
    m.speed()
    return m
