import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src import *
from src.util import grassmannian_dist_chordal
from src.timing import time_func, TimeAccumulator

def run_clustering_benchmark(N, K, n_points, n_centers, n_means, n_trials=3):
    ave_algos = dict(RGrAv=AsymptoticRGrAv(0.5), Flag=FlagMean(), Frechet=FrechetMeanByGradientDescent())
    dist_funcs = dict(RGrAv=grassmannian_dist_chordal, Flag=flagpole_distance, Frechet=grassmannian_dist_chordal)
    clustering_algos = dict()
    for ave_algo in ave_algos:
        if ave_algo == 'Flag':
            clustering_algos[ave_algo] = SubspaceClusteringFlagpole()
        else:
            clustering_algos[ave_algo] = SubspaceClustering(ave_algos[ave_algo], dist_funcs[ave_algo])

    timers = {algo: TimeAccumulator() for algo in ave_algos}
    performances = {algo: 0 for algo in ave_algos}

    for _ in range(n_trials):
        points, U_centers = generate_cluster_data(N, K, n_centers, n_points, center_dist=1.0, center_radius=0.1)
        
        for ave_algo in ave_algos:
            clusters = timers[ave_algo].time_func(clustering_algos[ave_algo].cluster, points, n_means)
            
            if ave_algo == 'Flag':
                dist_func = flagpole_subspace_distance
            else:
                dist_func = grassmannian_dist_chordal
            
            sum_squared_errors = 0
            for i in range(len(points)):
                min_dist = 1e9
                for j in range(len(clusters)):
                    dist = dist_func(points[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                sum_squared_errors += min_dist ** 2
            performances[ave_algo] += sum_squared_errors

    for algo in performances:
        performances[algo] /= n_trials

    return {algo: (timer.mean_time(), performances[algo]) for algo, timer in timers.items()}

def parameter_sweep(param_name, param_values, fixed_params):
    results = {algo: {'times': [], 'performances': []} for algo in ['RGrAv', 'Flag', 'Frechet']}
    
    for value in tqdm(param_values, desc=f"Sweeping {param_name}"):
        params = fixed_params.copy()
        params[param_name] = value
        benchmark_results = run_clustering_benchmark(**params)
        
        for algo in results:
            time, performance = benchmark_results[algo]
            results[algo]['times'].append(time)
            results[algo]['performances'].append(performance)
    
    return results

# Set up the fixed parameters and sweep ranges
fixed_params = {
    'N': 500,
    'K': 5,
    'n_points': 100,
    'n_centers': 10,
    'n_means': 20,
    'n_trials': 3,
}

sweep_params = {
    'N': np.linspace(100, 1000, 10, dtype=int),
    'K': np.arange(2, 11),
    'n_points': np.linspace(50, 500, 10, dtype=int),
    'n_centers': np.arange(5, 21),
    'n_means': np.arange(10, 41, 2)
}

# Perform parameter sweeps
all_results = {}
for param, values in sweep_params.items():
    all_results[param] = parameter_sweep(param, values, fixed_params)

# Plotting results
for param, values in sweep_params.items():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle(f'Parameter Sweep: {param}')
    
    for algo in ['RGrAv', 'Flag', 'Frechet']:
        ax1.plot(values, all_results[param][algo]['times'], label=algo)
        ax2.plot(values, all_results[param][algo]['performances'], label=algo)
    
    ax1.set_ylabel('Time (seconds)')
    ax1.legend()
    ax2.set_xlabel(param)
    ax2.set_ylabel('Sum of Squared Errors')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/parameter_sweep_{param}.png')
    plt.close()

print("Parameter sweeps completed. Results saved as PNG files.")
