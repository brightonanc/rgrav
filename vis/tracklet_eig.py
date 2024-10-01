# use tracklet data from SUMMET and cluster
# measure performance using cluster purity
# i.e., what % of the tracklets in a cluster are of the same class

import torch
import pickle
import numpy as np
from tqdm import tqdm

from src import *
from src.util import *
from src.timing import *

from data_sources.video_separation import SUMMET_Loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cdf(eigvals):
    xx = torch.linspace(0, 1, 1000)
    yy = [torch.sum(eigvals < x) / eigvals.numel() for x in xx]
    return xx, yy

# Load and preprocess data
summet_loader = SUMMET_Loader()
tracklets = summet_loader.tracklets
labels = summet_loader.labels
tracklets = tracklets.to(device)
tracklets -= tracklets.mean()

# take only walk tracklets
label_target = 'walk'
label_target = 'carry'
tracklets_trim = []
for i in range(tracklets.shape[0]):
    if labels[i] == label_target:
        tracklets_trim.append(tracklets[i])
tracklets = torch.stack(tracklets_trim, dim=0)

# trim to a few tracklets for testing
# tracklets = tracklets[:200]

tracklets_flat = tracklets.reshape(tracklets.shape[0], tracklets.shape[1], -1).permute(0, 2, 1)
print('tracklets_flat shape: ', tracklets_flat.shape)
points = []
for i in range(tracklets.shape[0]):
    tracklet_batch = tracklets_flat[i, :, :]
    U = torch.linalg.qr(tracklet_batch).Q
    points.append(U)
U_arr = torch.stack(points, dim=0)
print('U_arr shape: ', U_arr.shape)

n, d, k = U_arr.shape
P_ave = torch.zeros(d, d)
for i in tqdm(range(n), 'Adding projectors'):
    U = U_arr[i, :, :]
    P_ave += U @ U.T
P_ave /= n

eigvals, eigvecs = torch.linalg.eigh(P_ave)
plt.figure()
plt.subplot(121)
plt.plot(*cdf(eigvals))
plt.subplot(122)
plt.hist(eigvals, bins=100)
plt.show()

