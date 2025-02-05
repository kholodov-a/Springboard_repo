import numpy as np
import torch
import random

def seed_worker(worker_id):
    # SEED = 1812
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

