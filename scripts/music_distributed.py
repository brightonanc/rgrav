
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from subspace.algorithms import randn_complex

from src import AsymptoticDRGrAv, DeEPCA, ChebyshevConsensus, CycleGraph
from src.array_processing import generate_narrowband_weights_azel

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

N = 128
k = 1
orth = lambda A: torch.linalg.qr(A)[0]

# construct signal subspace
az = torch.tensor(0)
el = torch.tensor(0)
A = generate_narrowband_weights_azel(1, N, az, el).to(device)
A /= torch.norm(A)
az = torch.tensor(15)
el = torch.tensor(0)
A2 = generate_narrowband_weights_azel(1, N, az, el).to(device)
A2 /= torch.norm(A)
A = torch.cat([A, A2], dim=1); k = 2

def generate_sample(A, n, noise_level=1e-2):
    S = randn_complex(A.shape[1], n, device)
    E = randn_complex(N, n, device) * np.sqrt(noise_level)
    return A @ S + E

# ideal MUSIC
X = generate_sample(A, 10000)
R = X @ X.T.conj()

U_est, _, _ = torch.linalg.svd(R)
U_est = U_est[:, k:]

def music_spec(U, npts=100):
    azs = torch.linspace(-30, 30, npts)
    el = torch.tensor(0)
    spec = []
    for az in azs:
        a = generate_narrowband_weights_azel(1, N, az, el)
        denom = torch.norm(U.T.conj() @ a) ** 2
        spec.append(1 / denom.item())
    return azs, torch.tensor(spec)

azs, spec = music_spec(U_est)

plt.figure()
plt.plot(azs, spec)
plt.show()
