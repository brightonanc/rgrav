import torch
from tqdm import tqdm

from .rgrav import *
from .util import *

class RGravClustering:
    def __init__(self):
        self.rgrav = RGrAv()
        self.n_iters = 100

    def cluster(self, points, n_centers):
        # clustering pseudocode:
        # 1. initialize centers randomly
        # 2. assign points to centers
        # 3. update centers with RGrAv
        # 4. repeat until convergence
        import matplotlib.pyplot as plt
        losses = []
        U_centers = []
        # pick random points as initial centers
        for i in range(n_centers):
            ind = torch.randint(0, points.shape[0], (1,)).item()
            U_centers.append(points[ind])

        for i in tqdm(range(self.n_iters)):
            # do cluster assignment
            dists = torch.stack([torch.stack([grassmannian_dist(point, center) for center in U_centers]) for point in points])
            cluster_assignments = torch.argmin(dists, dim=1)

            # update centers
            center_changes = []
            for i in range(n_centers):
                cluster_points = points[cluster_assignments == i]
                if len(cluster_points) > 0:
                    new_center = self.rgrav.average(cluster_points)
                    center_changes.append(grassmannian_dist(U_centers[i], new_center))
                    U_centers[i] = new_center
                else:
                    center_changes.append(torch.tensor(0.0))

            # compute loss / check convergence
            loss = sum(center_changes)
            losses.append(loss)
            if len(losses) > 1 and losses[-1] >= losses[-2]:
                break

        return U_centers


def random_grassmann_point(N, K):
    U = torch.linalg.qr(torch.randn(N, K)).Q
    return U

def random_grassmann_tangent(U):
    N, K = U.shape
    H = torch.randn(N, K)
    H = H - U @ (U.T @ H)
    H /= torch.linalg.norm(H)
    return H

if __name__ == "__main__":
    N = 100; K = 10
    # N = 10; K = 3
    n_points = 100
    n_centers = 10
    center_dist = 0.5
    center_radius = 0.1
    center_dist = 1.0
    center_radius = 0.01

    # generate synthetic data
    U_center = random_grassmann_point(N, K)
    U_centers = []
    for _ in range(n_centers):
        H = random_grassmann_tangent(U_center)
        H *= center_dist
        U_centers.append(grassmannian_exp(U_center, H))

    pairwise_dists = []
    for i in range(n_centers):
        for j in range(i+1, n_centers):
            pairwise_dists.append(grassmannian_dist(U_centers[i], U_centers[j]))

    print(pairwise_dists)

    points = []
    for i in range(n_centers):
        Uc = U_centers[i]
        for _ in range(n_points):
            H = random_grassmann_tangent(Uc)
            H *= torch.rand(1).item() * center_radius
            point = grassmannian_exp(Uc, H)
            points.append(point)

    points = torch.stack(points, dim=0)
    print('points shape', points.shape)

    # do clustering
    rgrav_clustering = RGravClustering()
    clusters = rgrav_clustering.cluster(points, n_centers)

    closest_centers = []
    for cluster in clusters:
        dists = torch.stack([grassmannian_dist(Uc, cluster) for Uc in U_centers])
        closest_centers.append(torch.argmin(dists))

    cluster_dists = []
    for i in range(n_centers):
        cluster_dist = grassmannian_dist(U_centers[i], clusters[i])
        cluster_dists.append(cluster_dist)

    print('cluster distances', cluster_dists)
    print('closest centers', closest_centers)

    all_cluster_dists = []
    for i in range(n_centers):
        all_cluster_dists.append([])
        for j in range(n_centers):
            all_cluster_dists[i].append(grassmannian_dist(clusters[i], U_centers[j]))

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(all_cluster_dists)
    plt.colorbar()
    plt.show()
