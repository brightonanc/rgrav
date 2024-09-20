import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_sources.video_separation import Loader_CDW
from src import *
from src.util import *

if __name__ == '__main__':
    loader = Loader_CDW('highway')
    X = []
    for i in range(loader.n_samples):
        X.append(loader.load_data(i))
    X = np.stack(X, axis=0)
    print('data shape: ', X.shape)

    n_samples = X.shape[0]
    im_shape = X.shape[1:]
    n_dims = np.prod(im_shape)
    X = X.reshape(n_samples, n_dims).T
    print('data flattened: ', X.shape)

    X_batches = np.split(X, 100, axis=1)
    X_batches = X_batches[:10]
    print(X_batches[0].shape)

    U_arr = []
    for i in tqdm(range(len(X_batches))):
        X_batch = X_batches[i]
        U, S, Vt = np.linalg.svd(X_batch, full_matrices=False)
        U_arr.append(U)

    U_arr = torch.tensor(U_arr)

    rgrav = RGrAv()
    U_iters = []
    for iter_frame in tqdm(rgrav.algo_iters(U_arr)):
        U_est = iter_frame.U
        U_iters.append(U_est)
        if len(U_iters) > 30:
            break

    U_dists = [grassmannian_dist(U_iters[i], U_iters[i+1]) for i in range(len(U_iters)-1)]
    plt.figure()
    plt.plot(U_dists)

    final_U = U_iters[-1]
    for d in range(3):
        final_im = final_U[:, d].reshape(im_shape)
        final_im = (final_im - final_im.min()) / (final_im.max() - final_im.min())
        plt.figure()
        plt.imshow(final_im)
    plt.show()

