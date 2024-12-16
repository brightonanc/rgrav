
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

def generate_sample(A, n, noise_level=1e-1):
    S = randn_complex(A.shape[1], n, device)
    E = randn_complex(N, n, device) * np.sqrt(noise_level)
    return A @ S + E

# ideal MUSIC
X = generate_sample(A, 10000)
R = X @ X.T.conj()

U_est, _, _ = torch.linalg.svd(R)
U_est = U_est[:, k:]

def music_spec(U, npts=1000):
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
plt.plot(azs, spec, label='Centralized')

# Split data into 10 nodes
# each node contains noise subspace for local data
n_nodes = 10
node_samples = X.shape[1] // n_nodes
Us = []
for i in range(n_nodes):
    node_data = X[:, i*node_samples:(i+1)*node_samples]
    node_R = node_data @ node_data.T.conj()
    U, _, __ = torch.linalg.svd(node_R)
    U = U[:, k:]
    Us.append(U)
Us = torch.stack(Us)

# get average projector spectrum
P = 0.
for U in Us:
    P += U @ U.T.conj()
_, P_S, _ = torch.linalg.svd(P)

# do consensus with DRGrAv
max_iter = 10
consensus = ChebyshevConsensus(
    CycleGraph.get_positive_optimal_lapl_based_comm_W(n_nodes),
    cons_rounds = max_iter,
)

# Run RGrAv
kwargs = dict()
alpha = 0.1
rgrav = AsymptoticDRGrAv(alpha, consensus, **kwargs)
U_rgrav = rgrav.average(Us, max_iter=max_iter)
for i in range(U_rgrav.shape[0]):
    print('RGrAv dist', (N - k) - torch.norm(U_rgrav[0].T @ U_rgrav[i]) ** 2)
U_rgrav = U_rgrav[0]
# U_rgrav = U_rgrav[:, :k] + 1j * U_rgrav[:, k:]
U_rgrav = U_rgrav.to(torch.cfloat)

azs, spec = music_spec(U_rgrav)
plt.plot(azs, spec, label='DRGrAv')

# run DEEPCA
# deepca = DeEPCA(consensus) 
# U_deepca = deepca.average(Us, max_iter=max_iter)
# U_deepca = U_deepca[0]

# azs, spec = music_spec(U_deepca)
# plt.plot(azs, spec, label='DEEPCA')

plt.legend()

plt.figure()
plt.suptitle('Projector Spectrum')
plt.plot(P_S)

plt.show()
