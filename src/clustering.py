import torch
from tqdm import tqdm

from .rgrav import *
from .util import *

class SubspaceClustering:
    def __init__(self, ave_algo, dist_func):
        self.ave_algo = ave_algo
        self.dist_func = dist_func
        self.n_iters = 100
        self.center_tol = 1e-1

    def assign_clusters(self, points, centers):
        dists = torch.stack([torch.stack([self.dist_func(center, point) for center in centers]) for point in points])
        cluster_assignments = torch.argmin(dists, dim=1)
        return cluster_assignments

    def cluster(self, points, n_centers):
        # clustering pseudocode:
        # 1. initialize centers randomly
        # 2. assign points to centers with distance function
        # 3. update centers with clustering algorithm
        # 4. repeat until convergence

        changes = []
        U_centers = []
        # pick random points as initial centers
        for i in range(n_centers):
            ind = torch.randint(0, points.shape[0], (1,)).item()
            U_centers.append(points[ind])

        for i in tqdm(range(self.n_iters)):
            # if two centers are too close, reinit one
            for i in range(n_centers):
                for j in range(i+1, n_centers):
                    if self.dist_func(U_centers[i], U_centers[j]) < self.center_tol:
                        ind = torch.randint(0, points.shape[0], (1,)).item()
                        U_centers[i] = points[ind]

            # do cluster assignment
            cluster_assignments = self.assign_clusters(points, U_centers)

            # update centers
            center_changes = []
            for i in range(n_centers):
                cluster_points = points[cluster_assignments == i]
                if len(cluster_points) > 0:
                    new_center = self.ave_algo.average(cluster_points)
                else:
                    print('no points in cluster', i)
                    ind = torch.randint(0, points.shape[0], (1,)).item()
                    new_center = points[ind]
                center_changes.append(self.dist_func(U_centers[i], new_center))
                U_centers[i] = new_center

            # compute loss / check convergence
            change = sum(center_changes)
            changes.append(change)
            if len(changes) > 1 and changes[-1] >= changes[-2]:
                break

        return U_centers

def generate_cluster_data(N, K, n_centers, n_points, center_dist, center_radius):
    U_center = random_grassmannian_point(N, K)

    U_centers = []
    for _ in range(n_centers):
        H = random_grassmannian_tangent(U_center)
        H *= center_dist
        U_centers.append(grassmannian_exp(U_center, H))

    points = []
    for i in range(n_centers):
        Uc = U_centers[i]
        for _ in range(n_points):
            H = random_grassmannian_tangent(Uc)
            H *= torch.rand(1).item() * center_radius
            point = grassmannian_exp(Uc, H)
            points.append(point)

    points = torch.stack(points, dim=0)
    return points, U_centers
