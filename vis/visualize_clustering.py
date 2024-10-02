import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load the tracklet results
benchmark_results = pickle.load(open('benchmark_clustering_results.pkl', 'rb'))
sweep_params = benchmark_results['sweep_params']

# Prepare data for visualization
data = []
for sweep_type, sweep_data in benchmark_results.items():
    if sweep_type != 'sweep_params':
        sweep_values = sweep_params[sweep_type]
        for algorithm, results in sweep_data.items():
            for i in range(len(results['times'])):
                sweep_val = sweep_values[i]
                ave_time_val = results['times'][i].mean().item()
                ave_perf_val = results['performances'][i].mean().item()
                for j in range(len(results['times'][i])):
                    time_val = results['times'][i][j].item()
                    performance_val = results['performances'][i][j].item()
                    data.append({
                        'sweep_type': sweep_type,
                        'sweep_value': sweep_val,
                        'algorithm': algorithm,
                        'ave_time': ave_time_val,
                        'ave_perf': ave_perf_val,
                        'time': time_val,
                        'performance': performance_val,
                    })

df = pd.DataFrame(data)
try:
    plt.style.use('seaborn-whitegrid')
except:
    plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

# Create plots for each sweep type
for sweep_type in sweep_params.keys():
    # Filter data for the current sweep type
    sweep_df = df[df['sweep_type'] == sweep_type]
    
    # Plot sweep value vs time
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=sweep_df, x='sweep_value', y='time', hue='algorithm', marker='o')
    plt.title(f'{sweep_type.capitalize()} vs Time')
    plt.xlabel(sweep_type.capitalize())
    plt.ylabel('Time (seconds)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/{sweep_type}_vs_time.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot sweep value vs performance
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=sweep_df, x='sweep_value', y='performance', hue='algorithm', marker='o')
    plt.title(f'{sweep_type.capitalize()} vs Performance')
    plt.xlabel(sweep_type.capitalize())
    plt.ylabel('Sum of Squared Errors')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'plots/{sweep_type}_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Scatter plot of time vs performance
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(data=sweep_df, x='time', y='performance', hue='algorithm', size='sweep_value', sizes=(20, 200), alpha=0.7)
    plt.title(f'Time vs Performance ({sweep_type.capitalize()})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Sum of Squared Errors')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add annotations for sweep values
    for line in scatter.lines:
        for x, y, val in zip(line.get_xdata(), line.get_ydata(), sweep_df['sweep_value']):
            plt.annotate(f'{val}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'plots/{sweep_type}_time_vs_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create a heatmap of performance for all sweep types and algorithms
pivot_df = df.pivot_table(values='performance', index=['sweep_type', 'sweep_value'], columns='algorithm', aggfunc='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df, annot=True, fmt='.2e', cmap='YlGnBu', cbar_kws={'label': 'Performance (Sum of Squared Errors)'})
plt.title('Performance Heatmap')
plt.xlabel('Algorithm')
plt.ylabel('Sweep Type and Value')
plt.tight_layout()
plt.savefig('plots/performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
