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

# trim to a few tracklets for testing
tracklets = tracklets[:200]
labels = labels[:200]

unique_labels = set(labels)
n_labels = len(unique_labels)
print('unique labels: ', n_labels, unique_labels)

tracklets_flat = tracklets.reshape(tracklets.shape[0], tracklets.shape[1], -1).permute(2, 1, 0)

print('tracklets: ', tracklets.shape)
print('flat: ', tracklets_flat.shape)
print('labels: ', len(labels))

K = 24
n_subs = tracklets.shape[1] // K
assert n_subs * K == tracklets.shape[1]
n_tracklets = tracklets.shape[0]
points = []
U_labels = []

for i in range(n_subs):
    for j in range(n_tracklets):
        tracklet_batch = tracklets_flat[:, i * K:(i + 1) * K, j]
        U = torch.linalg.qr(tracklet_batch).Q
        points.append(U)
        U_labels.append(labels[j])
U_arr = torch.stack(points, dim=0)
print('U_arr: ', U_arr.shape)

clustering_algo = SubspaceClustering(AsymptoticRGrAv(0.5))
n_centers = n_labels
n_centers = 100
clusters = clustering_algo.cluster(U_arr, n_centers)

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
    print('cluster', i, 'label: ', cluster_labels[i])

cluster_purity = []
for i in range(len(clusters)):
    # find dominant label
    label_counts = {}
    total_count = 0
    for label in cluster_labels[i]:
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1
        total_count += 1

    if total_count == 0:
        cluster_purity.append(0)
    else:
        dominant_label = max(label_counts, key=label_counts.get)
        cluster_purity.append(label_counts[dominant_label] / total_count)

plt.figure()
plt.suptitle('Average Purity: {:.2f}'.format(sum(cluster_purity) / len(cluster_purity)))
plt.bar(list(cluster_labels.keys()), cluster_purity)

plt.figure()
plt.bar(list(cluster_labels.keys()), [len(cluster_labels[i]) for i in cluster_labels.keys()])

plt.figure()
plt.imshow(inter_cluster_dists)
plt.colorbar()
plt.show()

