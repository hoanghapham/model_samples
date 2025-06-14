import random
from numpy.typing import NDArray
import numpy as np

def init_centers(n_clusters: int, data: NDArray):
    centers = np.asarray(random.sample(list(data), k = n_clusters))
    centers = centers[np.argsort(centers[:, 0].ravel()), :]
    return centers

def euclidean_distance(point1: NDArray, point2: NDArray):
    assert point1.shape == point2.shape, "Two points must have the same dimensions"
    sum_of_squares = 0
    for coord1, coord2 in zip(point1, point2):
        sum_of_squares += (coord1 - coord2)**2
    dist = np.sqrt(sum_of_squares)
    return dist

def find_closest_center(point, centers: list):
    dist_to_centers = []

    # Calculate distance between the point to each center
    for center in centers:
        dist_to_centers.append(euclidean_distance(point, center))
    
    # Label range from 0 to n_clusters. 
    # Assign label to point by selecting the closest center.
    label = np.argmin(dist_to_centers)
    return label

class NaiveKMeans():
    def __init__(self, n_clusters: int = 2, n_max_iterations: int = 100, early_stop: bool = False) -> None:
        self.n_clusters = n_clusters
        self.n_max_iterations = n_max_iterations
        self.attributions_: list = []
        self.centers_: list = []
        self.early_stop = early_stop

    def _E(self, data):
        """Expectation step: Assign cluster label to points"""
        attributions = []
        cluster_stats = {label: 0 for label in range(self.n_clusters)}
        
        # Iterate through data points.
        for point in data:
            # Label range from 0 to n_clusters. 
            # Assign label to point by selecting the closest center.
            label = find_closest_center(point, self.centers_)
            attributions.append(label)

            # Keep track of number of points assigned to a cluster
            cluster_stats[label] = cluster_stats.get(label, 0) + 1

        # Randomly assign one point to a cluster without assignment
        re_indices = []

        for label, members in cluster_stats.items():
            if members == 0:
                re_idx = random.choice([i for i in range(len(data)) if i not in re_indices])
                attributions[re_idx] = label
                re_indices.append(re_idx)
        
        self.attributions_ = np.asarray(attributions)

    def _M(self, data):
        """Maximization step: Re-calculate centers by averaging all points in a cluster"""
        new_centers = []

        for label in range(self.n_clusters):
            center = data[np.array(self.attributions_) == label, :].mean(axis=0)
            new_centers.append(center)
        
        new_centers = np.asarray(new_centers)
        self.centers_ = new_centers

    def fit(self, X):
        """Run Expectation - Maximization n times until converge (no center movement)"""
        max_violations = 5
        violations = 0
        center_movements = [0]

        # Initiate first centers
        self.centers_ = init_centers(n_clusters=self.n_clusters, data=X)

        # Repeat Expectation - Maximization
        for _ in range(self.n_max_iterations):
            old_centers = self.centers_.copy()

            self._E(X)
            self._M(X)

            # Check if centers are still moving
            movement = sum([euclidean_distance(cen1, cen2) for cen1, cen2 in zip(old_centers, self.centers_)])
            center_movements.append(movement)
            delta = movement - center_movements[-1]

            if delta == 0:
                violations += 1
            else:
                violations = 0

            if violations == max_violations:
                # print("centers not moving, stop.")
                break
        
    def predict(self, X):
        attributions = []
        for point in X:
            label = find_closest_center(point, self.centers_)
            attributions.append(label)
        
        return attributions
