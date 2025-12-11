# This code generates Latin Hypercube samples for four parameters: NBI (integer), Ip, B0, and nbar.
# We use this to initialise the GPR training dataset.

import numpy as np
from scipy.stats import qmc

# Parameter ranges
nbi_min, nbi_max = 0, 40       # MW
Ip_min, Ip_max = 0.0, 5.0      # MA
B0_min, B0_max = 0.0, 4.0      # T
nbar_min, nbar_max = 0.0, 1.0  # Normalised density. Units: 10^20 m^-3

def sample_lhs(N, seed=None):
    """
    Generate N Latin Hypercube samples for (NBI, Ip, B0, nbar)
    Returns an array of shape (N, 4)
    """
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    U = sampler.random(N)  # shape (N, 4)

    # Scale each dimension
    U_scaled = np.zeros_like(U)

    # 1. NBI: integer range
    U_scaled[:, 0] = qmc.scale(U[:, [0]], nbi_min, nbi_max).flatten()
    U_scaled[:, 0] = np.round(U_scaled[:, 0]).astype(int)  # enforce integer

    # 2. Ip
    U_scaled[:, 1] = qmc.scale(U[:, [1]], Ip_min, Ip_max).flatten()

    # 3. B0
    U_scaled[:, 2] = qmc.scale(U[:, [2]], B0_min, B0_max).flatten()

    # 4. nbar
    U_scaled[:, 3] = qmc.scale(U[:, [3]], nbar_min, nbar_max).flatten()

    return U_scaled


# Create a batch of 20 samples
if __name__ == "__main__":
    X0 = sample_lhs(20, seed=0)
    print(X0)
