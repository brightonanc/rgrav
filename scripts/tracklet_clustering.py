# use tracklet data from SUMMET and cluster
# measure performance using cluster purity
# i.e., what % of the tracklets in a cluster are of the same class

import torch
import matplotlib.pyplot as plt

from src import *
from src.util import *
from src.timing import *

from data_sources.video_separation import SUMMET_Loader


summet_loader = SUMMET_Loader()
tracklets = summet_loader.tracklets
labels = summet_loader.labels
unique_labels = set(labels)
n_labels = len(unique_labels)
print('unique labels: ', n_labels, unique_labels)

# trim to a few tracklets for testing
# tracklets = tracklets[:100]

tracklets_flat = tracklets.reshape(tracklets.shape[0], tracklets.shape[1], -1).T
points = []
U_labels = []

K = 12
n_subs = tracklets.shape[1] // K
assert n_subs * K == tracklets.shape[1]

for i in range(n_subs):
    for j in range(tracklets.shape[-1]):
        tracklet_batch = tracklets_flat[:, i * K:(i + 1) * K, j]
        U = torch.linalg.qr(tracklet_batch).Q
        points.append(U)
        U_labels.append(labels[j])
U_arr = torch.stack(points, dim=0)
print('U_arr: ', U_arr.shape)

print('tracklets: ', tracklets.shape)
print('flat: ', tracklets_flat.shape)
print('labels: ', len(labels))

clustering_algo = SubspaceClustering(AsymptoticRGrAv(0.5))
clusters = clustering_algo.cluster(U_arr, n_centers=n_labels)

print('clusters: ', len(clusters), clusters[0].shape)

inter_cluster_dists = []
for i in range(len(clusters)):
    inter_cluster_dists.append([])
    for j in range(len(clusters)):
        if i != j:
            inter_cluster_dists[i].append(grassmannian_dist(clusters[i], clusters[j]))

cluster_assignments = clustering_algo.assign_clusters(U_arr, clusters)
cluster_labels = dict()
for i in range(len(clusters)):
    cluster_labels[i] = []
    for j in range(len(cluster_assignments)):
        if cluster_assignments[j] == i:
            cluster_labels[i].append(U_labels[j])

    cluster_labels[i] = set(cluster_labels[i])

plt.figure()
plt.bar(list(cluster_labels.keys()), [len(cluster_labels[i]) for i in cluster_labels.keys()])

plt.figure()
plt.imshow(inter_cluster_dists)
plt.colorbar()
plt.show()

