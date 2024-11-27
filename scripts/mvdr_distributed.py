
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
orth = lambda A: torch.linalg.qr(A)[0]

# construct signal subspace
az = torch.tensor(45)
el = torch.tensor(0)
A = generate_narrowband_weights_azel(1, N, az, el).to(device)
A /= torch.norm(A)
az = torch.tensor(-7)
el = torch.tensor(0)
A2 = generate_narrowband_weights_azel(1, N, az, el).to(device)
A2 /= torch.norm(A)
A = torch.cat([A, A2], dim=1); k = 2

def generate_sample(A, n, noise_level=3e-1):
    S = randn_complex(A.shape[1], n, device)
    E = randn_complex(N, n, device) * np.sqrt(noise_level)
    return A @ S + E

az_target = torch.tensor(30)
el_target = torch.tensor(0)
a = generate_narrowband_weights_azel(1, N, az_target, el_target).to(device)
a /= torch.norm(a)

# ideal MVDR
X = generate_sample(A, 10000)
R = X @ X.T.conj()
w_ideal = torch.linalg.solve(R, a)
w_ideal = w_ideal / (a.T.conj() @ w_ideal)
w_ideal = w_ideal / torch.norm(w_ideal)

def plot_beam(weights, npts=1024):
    angles = torch.linspace(-torch.pi, torch.pi, npts)
    angles = torch.arcsin(angles / torch.pi)
    angles = torch.rad2deg(angles)
    weight_spec = torch.fft.fft(weights, npts, dim=0)
    weight_spec = torch.fft.fftshift(weight_spec)
    weight_spec /= weight_spec.abs().max()
    plt.plot(angles, 10 * weight_spec.abs().cpu().log10())

n_subspaces = 10
Us = []
for _ in range(n_subspaces):
    X = generate_sample(A, 100)
    U, _, __ = torch.linalg.svd(X)
    U = U[:, :2]
    Us.append(U)

# total projection spectrum
P = 0.
for U in Us:
    P += U @ U.T.conj()
P /= len(Us)
_, S, _ = torch.linalg.svd(P)
plt.figure()
plt.suptitle('Average Projector Spectrum')
plt.hist(S[S>.5].cpu())
plt.hist(S[torch.logical_and(0<S, S<.5)].cpu())
# plt.show()


# Split data into 10 nodes
n_nodes = 10
node_samples = X.shape[1] // n_nodes
Us = []
for i in range(n_nodes):
    node_data = X[:, i*node_samples:(i+1)*node_samples]
    node_R = node_data @ node_data.T.conj()
    U = torch.linalg.eigh(node_R)[1][:, -k:]
    U = orth(torch.cat([U.real, U.imag], dim=1))
    Us.append(U)
Us = torch.stack(Us)

# Create line graph connectivity
consensus = ChebyshevConsensus(
    CycleGraph.get_positive_optimal_lapl_based_comm_W(n_nodes),
    cons_rounds = 8,
)

# Run RGrAv
kwargs = dict()
alpha = 0.1
rgrav = AsymptoticDRGrAv(alpha, consensus, **kwargs)
U_rgrav = rgrav.average(Us)
for i in range(U_rgrav.shape[0]):
    print('RGrAv dist', 4 - torch.norm(U_rgrav[0].T @ U_rgrav[i]) ** 2)
U_rgrav = U_rgrav[0]
U_rgrav = U_rgrav[:, :2] + 1j * U_rgrav[:, 2:]

R_rgrav = U_rgrav @ U_rgrav.T.conj()
w_rgrav = torch.linalg.solve(R_rgrav, a)
w_rgrav = w_rgrav / (a.T.conj() @ w_rgrav)
w_rgrav = w_rgrav / torch.norm(w_rgrav)

# Run DeEPCA
deepca = DeEPCA(consensus) 
U_deepca = deepca.average(Us)
U_deepca = deepca.average(Us)
for i in range(U_deepca.shape[0]):
    print('deepca dist', 4 - torch.norm(U_deepca[0].T @ U_deepca[i]) ** 2)
U_deepca = U_deepca[0]
U_deepca = U_deepca[:, :2] + 1j * U_deepca[:, 2:]
R_deepca = U_deepca @ U_deepca.T.conj()
w_deepca = torch.linalg.solve(R_deepca, a)
w_deepca = w_deepca / (a.T.conj() @ w_deepca)
w_deepca = w_deepca / torch.norm(w_deepca)


plt.figure()
plot_beam(w_ideal)
plot_beam(w_rgrav)
plot_beam(w_deepca)

plt.show()

