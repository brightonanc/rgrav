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

def run_clustering_benchmark(U_arr, labels, n_centers, n_trials=3):
    ave_algos = dict(RGrAv=AsymptoticRGrAv(0.05), Flag=FlagMean(), Frechet=FrechetMeanByGradientDescent(), Power=BPM())
    dist_funcs = dict(RGrAv=grassmannian_dist_chordal, Flag=flagpole_distance, Frechet=grassmannian_dist_chordal, Power=grassmannian_dist_chordal)

    clustering_algos = dict()
    for ave_algo in ave_algos:
        if ave_algo == 'Flag':
            clustering_algos[ave_algo] = SubspaceClusteringFlagpole()
        else:
            clustering_algos[ave_algo] = SubspaceClustering(ave_algos[ave_algo], dist_funcs[ave_algo])

    results = {algo: {'times': [], 'purities': []} for algo in ave_algos}

    for _ in range(n_trials):
        for ave_algo in ave_algos:
            start_time = time.time()
            clusters = clustering_algos[ave_algo].cluster(U_arr, n_centers)
            end_time = time.time()
            
            cluster_assignments = clustering_algos[ave_algo].assign_clusters(U_arr, clusters)
            
            cluster_labels = [[] for _ in range(n_centers)]
            for i, assignment in enumerate(cluster_assignments):
                cluster_labels[assignment].append(labels[i])
            
            cluster_purity = []
            for cluster in cluster_labels:
                if len(cluster) == 0:
                    cluster_purity.append(0)
                else:
                    label_counts = {}
                    for label in cluster:
                        label_counts[label] = label_counts.get(label, 0) + 1
                    dominant_label = max(label_counts, key=label_counts.get)
                    cluster_purity.append(label_counts[dominant_label] / len(cluster))
            
            results[ave_algo]['times'].append(end_time - start_time)
            results[ave_algo]['purities'].append(cluster_purity)

    return results

# Load and preprocess data
summet_loader = SUMMET_Loader()
tracklets = summet_loader.tracklets
labels = summet_loader.labels
tracklets = tracklets.to(device)

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

K = tracklets.shape[1]
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
print('U_arr: ', U_arr.shape, 'device: ', U_arr.device)

# Run benchmark with parameter sweep
n_centers_range = range(5, 55, 5)
n_trials = 3
# expedited run
# n_centers_range = range(10, 110, 10); n_trials = 1
all_results = {}

for n_centers in tqdm(n_centers_range, desc="Sweeping n_centers"):
    all_results[n_centers] = run_clustering_benchmark(U_arr, U_labels, n_centers, n_trials=n_trials)

# Save results
pickle.dump(all_results, open('tracklet_clustering_results.pkl', 'wb'))

print("Parameter sweep completed. Results saved as 'tracklet_clustering_results.pkl'.")
