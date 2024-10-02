# same as background_tracking.py, but with all metrics/plots


import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from data_sources.video_separation import Loader_CDW
from src import *
from src.util import *



datasets = ['canoe', 'highway']
dataset = 'canoe'
dataset = 'fountain01'
# dataset = 'fountain02'
loader = Loader_CDW(dataset)
X = []
for i in range(loader.n_samples):
    img = loader.load_data(i)
    X.append(img)
X = torch.stack(X, dim=0)
print('data shape: ', X.shape)

if dataset == 'canoe':
    X = X[800:]
elif dataset == 'highway':
    X = X[1300:]
elif 'fountain' in dataset:
    X = X[800:]
print('data shape: ', X.shape)

n_samples = X.shape[0]
im_shape = X.shape[1:]
n_dims = torch.prod(torch.tensor(im_shape))
X = X.reshape(n_samples, n_dims).T
imgs_mean = X.mean(dim=1)
print('data flattened: ', X.shape)

K = 10
n_split = n_samples // K
n_samples = n_split * K
X = X[:, :n_samples]
X_batches = torch.split(X, K, dim=1)
print('single batch shape: ',X_batches[0].shape)

# Benchmark batch SVDs + RGrAv
start_time = time.time()
U_arr = []
for i in tqdm(range(len(X_batches)), desc='Batch SVDs'):
    X_batch = X_batches[i]
    U, S, Vt = torch.linalg.svd(X_batch, full_matrices=False)
    U_arr.append(U)
U_arr = torch.stack(U_arr, dim=0)

U_aves = dict()
rgrav = AsymptoticRGrAv(0.5)
U_aves['RGrAv'] = rgrav.average(U_arr)
rgrav_time = time.time() - start_time

# Benchmark GRASTA
start_time = time.time()
grasta_losses = []
grasta = GRASTA(n_dims, K, C=1e-1)
observation_fraction = 0.3
for n in tqdm(range(100), 'Running GRASTA'):
    sample = X[:, n:n+1].cpu().numpy() / 100
    omega = np.arange(n_dims)
    np.random.choice(n, int(observation_fraction * n), replace=False)
    v_omega = sample[omega]
    grasta.add_data(v_omega, omega)
    grasta_U = torch.from_numpy(grasta.U).float()
    grasta_losses.append(grassmannian_dist_chordal(grasta_U, U_aves['RGrAv']))
U_aves['GRASTA'] = torch.from_numpy(grasta.U).float()
grasta_time = time.time() - start_time

# Print benchmark results
print("\nBenchmark Results:")
print(f"{'Method':<20}{'Time (seconds)':<20}")
print("-" * 40)
print(f"{'Batch SVDs + RGrAv':<20}{rgrav_time:<20.2f}")
print(f"{'GRASTA':<20}{grasta_time:<20.2f}")

plt.figure()
plt.title('GRASTA Loss')
plt.plot(grasta_losses)

if not os.path.exists('plots'):
    os.makedirs('plots')

# do background subtraction and plot video
all_img_backs = dict()
all_img_fores = dict()
for ave_type in U_aves:
    img_backs = []
    img_fores = []
    U_ave = U_aves[ave_type]
    for i in range(n_samples):
        img_flat = X[:, i]
        img = img_flat.view(im_shape)
        img_back = U_ave @ (U_ave.T @ (img_flat - imgs_mean)) + imgs_mean
        img_fore = img_flat - img_back
        
        img_back = img_back.view(im_shape)
        img_fore = img_fore.view(im_shape)
        img_backs.append(img_back)
        img_fores.append(img_fore)

    img_backs = torch.stack(img_backs, dim=0)
    img_fores = torch.stack(img_fores, dim=0)
    print('img fores', img_fores.min(), img_fores.max(), img_fores.mean(), img_fores.std())
    print('img backs', img_backs.min(), img_backs.max(), img_backs.mean(), img_backs.std())
    img_fores += 0.2
    img_backs = torch.clip(img_backs, 0, 1)
    img_fores = torch.clip(img_fores, 0, 1)
    all_img_backs[ave_type] = img_backs
    all_img_fores[ave_type] = img_fores

frame_nums = range(0, n_samples, 100)
for frame in frame_nums:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('Original')
    plt.imshow(X[:, frame].view(im_shape))
    for a, ave_type in enumerate(U_aves):
        img_fore = all_img_fores[ave_type][frame]
        plt.subplot(1, 3, a+2)
        plt.title(ave_type)
        plt.imshow(img_fore)
    plt.savefig(f'plots/foreground_{dataset}_{frame}.png')
    plt.close()

# plt.show()
