import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load the tracklet results
tracklet_results = pickle.load(open('tracklet_clustering_results.pkl', 'rb'))

# Extract data for visualization
n_clusters = list(tracklet_results.keys())
algorithms = list(tracklet_results[n_clusters[0]].keys())
print(n_clusters, algorithms)

# Prepare data for visualization
data = []
for n in n_clusters:
    for algo in algorithms:
        runtimes = tracklet_results[n][algo]['times']
        purities = tracklet_results[n][algo]['purities']
        ave_purities = [sum(pts) / len(pts) for pts in purities]
        for runtime, purity in zip(runtimes, ave_purities):
            data.append({
                'n_clusters': n,
                'algorithm': algo,
                'runtime': runtime,
                'purity': purity
            })

df = pd.DataFrame(data)

# Visualization 1: Box plot of runtimes
plt.figure(figsize=(12, 6))
sns.boxplot(x='n_clusters', y='runtime', hue='algorithm', data=df)
plt.title('Runtime Comparison of Clustering Algorithms')
plt.xlabel('Number of Clusters')
plt.ylabel('Runtime (seconds)')
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('runtime_comparison.png')
plt.close()

# Visualization 1: Box plot of purities
plt.figure(figsize=(12, 6))
sns.boxplot(x='n_clusters', y='purity', hue='algorithm', data=df)
plt.title('Runtime Comparison of Clustering Algorithms')
plt.xlabel('Number of Clusters')
plt.ylabel('Runtime (seconds)')
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('purity_comparison.png')
plt.close()

# Visualization 2: Heatmap of average cluster purity
pivot_purity = df.pivot_table(values='purity', index='algorithm', columns='n_clusters', aggfunc='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_purity, annot=True, cmap='YlGnBu', fmt='.3f')
plt.title('Average Cluster Purity')
plt.xlabel('Number of Clusters')
plt.ylabel('Algorithm')
plt.tight_layout()
plt.savefig('purity_heatmap.png')
plt.close()

# Visualization 3: Scatter plot of runtime vs cluster purity
plt.figure(figsize=(12, 8))
sns.scatterplot(x='runtime', y='purity', hue='algorithm', size='n_clusters', data=df, sizes=(20, 200))
plt.title('Runtime vs Cluster Purity')
plt.xlabel('Runtime (seconds)')
plt.ylabel('Cluster Purity')
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('runtime_vs_purity.png')
plt.close()

# Visualization 4: Line plot of average runtime vs number of clusters
pivot_runtime = df.pivot_table(values='runtime', index='n_clusters', columns='algorithm', aggfunc='mean')
plt.figure(figsize=(12, 6))
sns.lineplot(data=pivot_runtime)
plt.title('Average Runtime vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Runtime (seconds)')
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('avg_runtime_vs_clusters.png')
plt.close()

# Visualization 4: Line plot of average runtime vs number of clusters
pivot_purity = df.pivot_table(values='purity', index='n_clusters', columns='algorithm', aggfunc='mean')
plt.figure(figsize=(12, 6))
sns.lineplot(data=pivot_purity)
plt.title('Average Purity vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Average Cluster Purity')
plt.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('avg_purity_vs_clusters.png')
plt.close()
