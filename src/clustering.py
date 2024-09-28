import torch
from tqdm import tqdm

from .util import *
from .flag_mean import *

class SubspaceClustering:
    def __init__(self, ave_algo, dist_func):
        self.ave_algo = ave_algo
        self.dist_func = dist_func
        self.n_iters = 100
        self.center_tol = 1e-3

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

        for i in range(self.n_iters):
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
                    # print('no points in cluster', i)
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

class SubspaceClusteringFlagpole:
    '''
    This is slightly different because flagpole vs subspace dist
    Also initialization has to be different
    '''
    def __init__(self):
        self.ave_algo = FlagMean()
        self.n_iters = 100
        self.center_tol = 1e-3

    def assign_clusters(self, points, centers):
        dists = torch.stack([torch.stack([flagpole_subspace_distance(center, point) for center in centers]) for point in points])
        cluster_assignments = torch.argmin(dists, dim=1)
        return cluster_assignments

    def get_random_flagpole_batch(self, points):
        inds = torch.randint(0, points.shape[0], (20,))
        batch = [points[ind.item()] for ind in inds]
        batch = torch.stack(batch, dim=0)
        rand_flagpole = self.ave_algo.average(batch)
        return rand_flagpole

    def get_random_flagpole(self, points):
        ind = torch.randint(0, points.shape[0], (1,)).item()
        rand_flagpole = get_flagpole(points[ind])
        return rand_flagpole

    def cluster(self, points, n_centers):
        # clustering pseudocode:
        # 1. initialize centers randomly
        # 2. assign points to centers with distance function
        # 3. update centers with clustering algorithm
        # 4. repeat until convergence

        changes = []
        U_centers = []
        # pick a few random points and flagpole them for initial centers
        for i in range(n_centers):
            U_centers.append(self.get_random_flagpole(points))

        for i in range(self.n_iters):
            # if two centers are too close, reinit one
            for i in range(n_centers):
                for j in range(i+1, n_centers):
                    if flagpole_distance(U_centers[i], U_centers[j]) < self.center_tol:
                        U_centers[i] = self.get_random_flagpole(points)

            # do cluster assignment
            cluster_assignments = self.assign_clusters(points, U_centers)

            # update centers
            center_changes = []
            for i in range(n_centers):
                cluster_points = points[cluster_assignments == i]
                if len(cluster_points) > 0:
                    new_center = self.ave_algo.average(cluster_points)
                else:
                    # print('no points in cluster', i)
                    new_center = self.get_random_flagpole(points)
                center_changes.append(flagpole_distance(U_centers[i], new_center))
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
