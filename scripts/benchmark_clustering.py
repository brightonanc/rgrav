import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src import *
from src.util import grassmannian_dist_chordal
from src.timing import time_func, TimeAccumulator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_clustering_benchmark(N, K, n_points, n_centers, n_means, n_trials=3):
    ave_algos = dict(RGrAv=AsymptoticRGrAv(0.1), Flag=FlagMean(), Frechet=FrechetMeanByGradientDescent())
    dist_funcs = dict(RGrAv=grassmannian_dist_chordal, Flag=flagpole_subspace_distance, Frechet=grassmannian_dist_chordal)
    clustering_algos = dict()
    for ave_algo in ave_algos:
        if ave_algo == 'Flag':
            clustering_algos[ave_algo] = SubspaceClusteringFlagpole()
        else:
            clustering_algos[ave_algo] = SubspaceClustering(ave_algos[ave_algo], dist_funcs[ave_algo])

    timers = {algo: TimeAccumulator() for algo in ave_algos}
    performances = {algo: [] for algo in ave_algos}

    for _ in range(n_trials):
        points, U_centers = generate_cluster_data(N, K, n_centers, n_points, center_dist=1.0, center_radius=0.1)
        points = points.to(device)
        U_centers = torch.stack(U_centers, dim=0).to(device)
        
        for ave_algo in ave_algos:
            clusters = timers[ave_algo].time_func(clustering_algos[ave_algo].cluster, points, n_means)
            dist_func = dist_funcs[ave_algo]
            
            sum_squared_errors = 0
            for i in range(len(points)):
                min_dist = float('inf')
                for j in range(len(clusters)):
                    dist = dist_func(points[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                sum_squared_errors += min_dist ** 2
            performances[ave_algo].append(sum_squared_errors)

    return {algo: (timer.times, performances[algo]) for algo, timer in timers.items()}

def parameter_sweep(param_name, param_values, fixed_params):
    results = dict()
    
    for value in tqdm(param_values, desc=f"Sweeping {param_name}"):
        params = fixed_params.copy()
        params[param_name] = value
        benchmark_results = run_clustering_benchmark(**params)
        
        for algo in benchmark_results:
            time, performance = benchmark_results[algo]
            if algo not in results:
                results[algo] = {'times': [], 'performances': []}
            results[algo]['times'].append(time)
            results[algo]['performances'].append(performance)

    for algo in results:
        results[algo]['times'] = torch.tensor(results[algo]['times'])
        results[algo]['performances'] = torch.tensor(results[algo]['performances'])
    
    return results

# Set up the fixed parameters and sweep ranges
fixed_params = {
    'N': 1000,
    'K': 50,
    'n_points': 100,
    'n_centers': 10,
    'n_means': 20,
    'n_trials': 5,
}

sweep_params = {
    'N': [100, 200, 500, 1000],
    'K': [5, 10, 20, 50, 100],
    # 'n_points': np.linspace(10, 100, 10, dtype=int),
    # 'n_centers': np.arange(5, 11),
    # 'n_means': np.arange(10, 21, 2)
}

sweep_params = {
    'N': [1000, 500, 200, 100],
}
sweep_params = {
    'K': [100, 50, 20, 10],
}
sweep_params = {
    'n_points': np.geomspace(10, 100, 5, dtype=int)[::-1],
}
sweep_params = {
    'N': [1000, 500, 200, 100],
    'K': [100, 50, 20, 10, 5],
    'n_points': np.geomspace(10, 500, 10, dtype=int)[::-1],
}
sweep_params = {
    'K': np.geomspace(5, 100, 10, dtype=int)[::-1],
    'n_points': np.geomspace(10, 500, 10, dtype=int)[::-1],
}

# Perform parameter sweeps
all_results = {}
for param, values in sweep_params.items():
    all_results[param] = parameter_sweep(param, values, fixed_params)
pickle.dump(all_results, open('benchmark_clustering_results.pkl', 'wb'))

# Plotting results
for param, values in sweep_params.items():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle(f'Parameter Sweep: {param}')
    
    for algo in all_results[param]:
        if 'Frechet' in algo:
            algo_label = 'Fréchet'
        else:
            algo_label = algo
        ax1.plot(values, all_results[param][algo]['times'].cpu().numpy(), label=algo_label)
        ax2.plot(values, all_results[param][algo]['performances'].cpu().numpy(), label=algo_label)
    
    ax1.set_ylabel('Time (seconds)')
    ax1.legend()
    ax2.set_xlabel(param)
    ax2.set_ylabel('Sum of Squared Errors')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/parameter_sweep_{param}.png')
    plt.close()

print("Parameter sweeps completed. Results saved as PNG files.")
