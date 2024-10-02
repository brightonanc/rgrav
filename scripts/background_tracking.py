import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_sources.video_separation import Loader_CDW
from src import *
from src.util import *

if __name__ == '__main__':
    dataset = 'canoe'
    dataset = 'highway'
    loader = Loader_CDW(dataset)
    X = []
    for i in range(loader.n_samples):
        img = loader.load_data(i)
        X.append(img)
    X = torch.stack(X, dim=0)
    print('data shape: ', X.shape)

    if dataset == 'canoe':
        X = X[800:]
        print('data shape: ', X.shape)
    elif dataset == 'highway':
        X = X[1300:]
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
    X_batches = torch.split(X, n_split, dim=1)
    print('single batch shape: ',X_batches[0].shape)

    U_arr = []
    for i in tqdm(range(len(X_batches))):
        X_batch = X_batches[i]
        U, S, Vt = torch.linalg.svd(X_batch, full_matrices=False)
        U_arr.append(U)
    U_arr = torch.stack(U_arr, dim=0)

    # compute average projector
    # jk it's too large of a matrix
    # P = 0.
    # for U in U_arr:
    #     U = U[::10]
    #     P += U @ U.T
    # P /= len(U_arr)
    # P_eigs = torch.linalg.eigvalsh(P)
    # plt.figure()
    # plt.plot(P_eigs)
    # plt.show()

    # visualize first few Us
    plt.figure()
    plt.suptitle('First 3 Components of First 3 Us')
    for m in range(3):
        for d in range(3):
            im = U_arr[m][:, d].reshape(im_shape)
            im = (im - im.min()) / (im.max() - im.min())
            plt.subplot(331 + m * 3 + d)
            plt.imshow(im)

    rgrav = AsymptoticRGrAv(0.5)
    U_iters = []
    for iter_frame in tqdm(rgrav.algo_iters(U_arr)):
        U_est = iter_frame.U
        U_iters.append(U_est)
        if len(U_iters) > 50:
            break

    U_dists = [grassmannian_dist(U_iters[i], U_iters[i+1]) for i in range(len(U_iters)-1)]
    U_dists0 = [grassmannian_dist(U_iters[0], U_iters[i]) for i in range(len(U_iters))]
    plt.figure()
    plt.subplot(121)
    plt.plot(U_dists)
    plt.subplot(122)
    plt.plot(U_dists0)

    final_U = U_iters[-1]
    # compute approximate eigenvalues
    evals = torch.linalg.norm(final_U.T @ X, dim=1) ** 2
    plt.figure()
    plt.suptitle('Eigenvalues of data using final U')
    plt.plot(evals)

    for d in range(3):
        final_im = final_U[:, d].reshape(im_shape)
        final_im = (final_im - final_im.min()) / (final_im.max() - final_im.min())
        plt.figure()
        plt.imshow(final_im)

    if not os.path.exists('plots'):
        os.makedirs('plots')

    # do background subtraction and plot video
    img_backs = []
    img_fores = []
    for i in range(n_samples):
        img_flat = X[:, i]
        img = img_flat.view(im_shape)
        img_back = final_U @ (final_U.T @ (img_flat - imgs_mean)) + imgs_mean
        img_fore = img_flat - img_back
        
        img_back = img_back.view(im_shape)
        img_fore = img_fore.view(im_shape)
        img_backs.append(img_back)
        img_fores.append(img_fore)

    # normalize all images from foreground and background
    img_fores = torch.stack(img_fores, dim=0)
    img_backs = torch.stack(img_backs, dim=0)
    print('img fores', img_fores.min(), img_fores.max(), img_fores.mean(), img_fores.std())
    print('img backs', img_backs.min(), img_backs.max(), img_backs.mean(), img_backs.std())
    img_fores += 0.2
    img_fores = torch.clip(img_fores, 0, 1)
    img_backs = torch.clip(img_backs, 0, 1)
    plt.figure()
    plt.subplot(121)
    plt.hist(img_fores.flatten()[:10000])
    plt.subplot(122)
    plt.hist(img_backs.flatten()[:10000])
    # plt.show()

    for i in range(n_samples):
        img = X[:, i].view(im_shape)
        img_back = img_backs[i]
        img_fore = img_fores[i]
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(img)
        plt.subplot(132)
        plt.imshow(img_back)
        plt.subplot(133)
        plt.imshow(img_fore)

        plt.savefig('plots/background_subtraction_{}.png'.format(i))
        plt.close()

    from src.gif import gif_folder
    gif_folder('plots', 'background_subtraction_')

    plt.show()

