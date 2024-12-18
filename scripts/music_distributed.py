
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src import AsymptoticDRGrAv, DeEPCA, ChebyshevConsensus, CycleGraph
from src.array_processing import generate_narrowband_weights_azel


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def randn_complex(a, b, device=device):
    ret = torch.randn(a, b, dtype=torch.cfloat, device=device) \
        + 1j * torch.randn(a, b, dtype=torch.cfloat, device=device)
    return ret

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

seed = 101
if seed:
    torch.manual_seed(seed)
    np.random.seed(seed)

N = 128
N = 32
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

k = 5
emitter_azs = torch.rand(k) * 60 - 30
As = []
for az in emitter_azs:
    As.append(generate_narrowband_weights_azel(1, N, az, el).to(device))
A = torch.cat(As, dim=1)
A = torch.linalg.qr(A).Q
U_true = torch.randn(N, N-k, dtype=torch.cfloat)
U_true = U_true - A @ (A.T.conj() @ U_true)
U_true = torch.linalg.qr(U_true)[0]

def generate_sample(A, n, noise_level=1e-1):
    S = randn_complex(A.shape[1], n, device)
    E = randn_complex(N, n, device) * np.sqrt(noise_level)
    return A @ S + E

# ideal MUSIC
X = generate_sample(A, 1000)
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
P /= n_nodes
_, P_S, _ = torch.linalg.svd(P)

# do consensus with DRGrAv
max_iter = 30
# max_iter = 5
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
    print('RGrAv dist', (N - k) - torch.norm(U_rgrav[0].T.conj() @ U_rgrav[i]) ** 2)
for i in range(U_rgrav.shape[0]):
    print('RGrAv dist data', (N - k) - torch.norm(Us[0].T.conj() @ U_rgrav[i]) ** 2)
U_rgrav = U_rgrav[0]
# U_rgrav = U_rgrav[:, :k] + 1j * U_rgrav[:, k:]
U_rgrav = U_rgrav.to(torch.cfloat)

azs, spec = music_spec(U_rgrav)
plt.figure()
plt.plot(azs, spec, label='DRGrAv')

# run DEEPCA
deepca = DeEPCA(consensus) 
U_deepca = deepca.average(Us, max_iter=max_iter)
for i in range(U_deepca.shape[0]):
    print('DEEPCA dist', (N - k) - torch.norm(U_deepca[0].T.conj() @ U_deepca[i]) ** 2)
for i in range(U_deepca.shape[0]):
    print('DEEPCA dist data', (N - k) - torch.norm(Us[0].T.conj() @ U_deepca[i]) ** 2)
U_deepca = U_deepca[0]

azs, spec = music_spec(U_deepca)
plt.plot(azs, spec, label='DEEPCA')

plt.legend()
for az in emitter_azs:
    plt.axvline(az, c='k', alpha=0.7, linestyle='-')

plt.figure()
plt.suptitle('Projector Spectrum')
plt.plot(P_S)

U_rgravs = []
max_iters = range(5, 35, 5)
# max_iters = range(10, 110, 20)
# max_iters = range(1, 6)
for max_iter in max_iters:
    consensus = ChebyshevConsensus(
        CycleGraph.get_positive_optimal_lapl_based_comm_W(n_nodes),
        cons_rounds = max_iter,
    )
    rgrav = AsymptoticDRGrAv(alpha, consensus, **kwargs)
    U_rgrav = rgrav.average(Us, max_iter=max_iter)
    U_rgrav = U_rgrav[0]
    U_rgravs.append(U_rgrav)

plt.figure()
for i in range(len(max_iters)):
    azs, spec = music_spec(U_rgravs[i])
    plt.plot(azs, spec, label=f'DRGrAv i={max_iters[i]}')
plt.legend()

U_deepcas = []
for max_iter in max_iters:
    consensus = ChebyshevConsensus(
        CycleGraph.get_positive_optimal_lapl_based_comm_W(n_nodes),
        cons_rounds = max_iter,
    )
    deepca = DeEPCA(consensus)
    U_deepca = deepca.average(Us, max_iter=max_iter)
    U_deepca = U_deepca[0]
    U_deepcas.append(U_deepca)

plt.figure()
for i in range(len(max_iters)):
    azs, spec = music_spec(U_deepcas[i])
    plt.plot(azs, spec, label=f'DEEPCA i={max_iters[i]}')
plt.legend()

def subspace_dist(U, V):
    assert U.shape == V.shape, f'{U.shape}, {V.shape}'
    assert U.shape[1] == N - k
    return (N - k) - torch.norm(U.T.conj() @ V) ** 2

sub_errs = dict()
sub_errs['DRGrAv'] = []
for U_rgrav in U_rgravs:
    sub_errs['DRGrAv'].append(subspace_dist(U_true, U_rgrav))
sub_errs['DEEPCA'] = []
for U_deepca in U_deepcas:
    sub_errs['DEEPCA'].append(subspace_dist(U_true, U_deepca))

plt.figure()
for k, v in sub_errs.items():
    plt.plot(max_iters, v, label=k)
plt.legend()

plt.show()
