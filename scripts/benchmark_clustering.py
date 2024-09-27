import torch
import matplotlib.pyplot as plt

from src import *
from src.util import grassmannian_dist_chordal
from src.timing import time_func, TimeAccumulator


N = 100
K = 10
n_points = 100
n_centers = 10
center_dist = 1.0
center_radius = 0.1
points, U_centers = generate_cluster_data(N, K, n_centers, n_points, center_dist, center_radius)

# do clustering
ave_algos = dict(RGrAv=AsymptoticRGrAv(0.5), Flag=FlagMean(), Frechet=FrechetMeanByGradientDescent())
dist_funcs = dict(RGrAv=grassmannian_dist_chordal, Flag=flagpole_distance, Frechet=grassmannian_dist_chordal)
clustering_algos = dict()
for ave_algo in ave_algos:
    if ave_algo == 'Flag':
        clustering_algos[ave_algo] = SubspaceClusteringFlagpole()
    else:
        clustering_algos[ave_algo] = SubspaceClustering(ave_algos[ave_algo], dist_funcs[ave_algo])

clusters = dict()
timers = dict()
for ave_algo in ave_algos:
    print('running clustering with', ave_algo)
    timers[ave_algo] = TimeAccumulator()
    clusters[ave_algo] = timers[ave_algo].time_func(clustering_algos[ave_algo].cluster, points, n_centers)

print("\nClusteringBenchmark Results:")
print(f"{'Method':<20}{'Time (seconds)':<20}")
print("-" * 40)
for ave_algo, timer in timers.items():
    print(f"{ave_algo:<20}{timer.mean_time():<20.2f}")
print()
exit()

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
