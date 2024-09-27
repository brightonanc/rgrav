import torch
import matplotlib.pyplot as plt

from src import *
from src.util import grassmannian_dist_chordal
from src.timing import time_func, TimeAccumulator


N = 100; K = 10
# N = 10; K = 3
n_points = 100
n_centers = 10
center_dist = 1.0
center_radius = 0.1
points, U_centers = generate_cluster_data(N, K, n_centers, n_points, center_dist, center_radius)

# do clustering
ave_algos = [AsymptoticRGrAv(0.5), FlagMean(), FrechetMeanByGradientDescent()]
for ave_algo in ave_algos:
    clustering_algo = SubspaceClustering(ave_algo)
clusters = clustering_algo.cluster(points, n_centers)

closest_centers = []
for cluster in clusters:
    dists = torch.stack([grassmannian_dist_chordal(Uc, cluster) for Uc in U_centers])
    closest_centers.append(torch.argmin(dists))

cluster_dists = []
for i in range(n_centers):
    cluster_dist = grassmannian_dist_chordal(U_centers[i], clusters[i])
    cluster_dists.append(cluster_dist)

print('cluster distances', cluster_dists)
print('closest centers', closest_centers)

all_cluster_dists = []
for i in range(n_centers):
    all_cluster_dists.append([])
    for j in range(n_centers):
        all_cluster_dists[i].append(grassmannian_dist_chordal(clusters[i], U_centers[j]))

plt.figure()
plt.imshow(all_cluster_dists)
plt.colorbar()
plt.show()
