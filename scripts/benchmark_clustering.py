import torch
import matplotlib.pyplot as plt

from src import *
from src.util import grassmannian_dist_chordal
from src.timing import time_func, TimeAccumulator


N = 500
K = 5
n_points = 100
n_centers = 10
n_means = n_centers * 2
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
    clusters[ave_algo] = timers[ave_algo].time_func(clustering_algos[ave_algo].cluster, points, n_means)

print("\nClusteringBenchmark Results:")
print(f"{'Method':<20}{'Time (seconds)':<20}")
print("-" * 40)
for ave_algo, timer in timers.items():
    print(f"{ave_algo:<20}{timer.mean_time():<20.2f}")
print()

all_cluster_dists = dict()
for ave_algo in ave_algos:
    all_cluster_dists[ave_algo] = []
    for i in range(n_means):
        all_cluster_dists[ave_algo].append([])
        for j in range(n_centers):
            if ave_algo == 'Flag':
                all_cluster_dists[ave_algo][i].append(flagpole_subspace_distance(clusters[ave_algo][i], U_centers[j]))
            else:
                all_cluster_dists[ave_algo][i].append(grassmannian_dist_chordal(clusters[ave_algo][i], U_centers[j]))

plt.figure()
for m, ave_algo in enumerate(ave_algos):
    plt.subplot(131 + m)
    plt.title(ave_algo)
    plt.imshow(all_cluster_dists[ave_algo], aspect='auto')
    plt.colorbar()

# compute sum of squared errors for all points and all algorithms
for ave_algo in ave_algos:
    if ave_algo == 'Flag':
        dist_func = flagpole_subspace_distance
    else:
        dist_func = grassmannian_dist_chordal
    sum_squared_errors = 0
    for i in range(len(points)):
        min_dist = 1e9
        for j in range(len(clusters[ave_algo])):
            dist = dist_func(points[i], clusters[ave_algo][j])
            if dist < min_dist:
                min_dist = dist
        sum_squared_errors += min_dist ** 2
    print(f"{ave_algo}: {sum_squared_errors}")

plt.show()
