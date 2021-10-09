#!/usr/bin/env python
import numpy as np
from collections import Counter
from tqdm import tqdm
"""
Zhao Shuyang, contact@zhaoshuyang.com
k-medoid clustering usin partition around medoid (PAM) algorithm. Maybe FasterPAM will be added in the future.
It takes time to digest the paper.
"""


def farthest_search(dist_mat, k):
    """
    Farthest-first traversal:
    Spaning the data points by having adding farthest data points to the current set until the cardinality reaches k.
    """
    N,N = dist_mat.shape
    init_medoid = np.random.randint(N)
    medoids = [init_medoid]
    print ("Farthest-first traversal of {0} over {1} samples".format(k, N))
    for i in tqdm(range(1, k)):
        dist = np.zeros((N))
        for j in range(N):
            dist[j] = np.min(dist_mat[j][np.array(medoids)])
        medoids.append(np.argmax(dist))
        
    return np.array(medoids)


class KMedoidClustering():
    def __init__(self, dist_mat, K):
        self.dist_mat = dist_mat  # Distancs matrix
        self.K = int(K)
        self.cluster_algorithm = PAM()
        self.clusters = {}  # key as medoid and value as list of cluster members
        self.medoids = []  # Medoids as a list

    def perform_clustering(self):
        self.medoids, _nearest_medoids = self.cluster_algorithm.run(self.dist_mat, self.K)
        for i in range(len(self.medoids)):
            curr_cluster = np.where(_nearest_medoids == self.medoids[i])[0]
            self.clusters[self.medoids[i]] = curr_cluster.astype(np.int)

    def repartition(self):
        distances_to_medoids = self.dist_mat[:, self.medoids]
        nearest_medoid = self.medoids[np.argmin(distances_to_medoids, axis=1)]
        nearest_medoid[self.medoids] = self.medoids # Nearest medoid for the medoids are themselvse
        
        #_nearest_medoids = self.cluster_algorithm.assign_points_to_clusters(np.array(self.medoids), self.dist_mat)
        for i in range(len(self.medoids)):
            curr_cluster = np.where(nearest_medoid == self.medoids[i])[0]
            self.clusters[self.medoids[i]] = curr_cluster.astype(np.int)
            
    def sort_clusters(self, order_by='size'):
        if order_by == 'size':
            medoid_size_tuple = sorted([(len(self.clusters[m]), np.random.random(), m) for m in self.medoids])
            self.medoids = [m for m in reversed([m for (tmp1, tmp2, m) in medoid_size_tuple])]

        if order_by == 'distance':
            medoid_dist_mat = self.dist_mat[self.medoids][:,self.medoids]
            medoid_pmt = farthest_search(medoid_dist_mat, len(self.medoids))
            self.medoids = self.medoids[medoid_pmt].tolist()

        if order_by == 'random':
            self.medoids = self.medoids[np.random.permutation(len(self.medoids))]

            
    def purity(self, item_classIDs):
        #A list indexed the same as the distance matrix
        n_majority, n = 0, 0
        for medoid in self.medoids:
            cluster = self.clusters[medoid]
            membership = np.array(item_classIDs)[cluster]
            c = Counter(membership)
            n_majority += c.most_common(1)[0][1]
            n += len(membership)
        try:
            return n_majority*1./n
        except:
            print ("Purity calculation fails, {0}, {1}.".format(n_majority,n))
            return None

    def evaluate_cluster_tightness(self):
        total_dist = 0
        max_dist = 0
        for medoid, cluster in iter(self.clusters.items()):
            dist = np.max(self.dist_mat[medoid][cluster])
            max_dist = max_dist < dist and dist or max_dist
            total_dist += np.sum(self.dist_mat[medoid][cluster])
        print ("Total distantces: {0}, max nearest medoid distance {1}".format(total_dist, max_dist))
        return
            
    def show(self):
        for medoid in self.medoids:
            print("medoid:{0}".format(medoid))
            cluster = self.clusters[medoid]
            print("cluster members: {0}".format(cluster))


class PAM():
    def __init__(self):
        pass
        
    def run(self, dist_mat, k, init='farthest'):
        """
        Perform K medoids clustering based on a distance matrix. Initialize the medoids using farthest search
        Args
        - dist_mat: the distance matrix
        - k: the number of clusters to find
        Outputs:
        - curr_medoids: the obtained medoids, represented by a list of indices as in the distance matrix.
        - nearest_medoid_idx: Each cluster is represented in a list. Each cluster member is represented as its index in the distance matrix. 
        """

        if init == 'farthest':
            curr_medoids = farthest_search(dist_mat, k)
        elif init == 'random':
            curr_medoids = np.random.permutation(len(dist_mat))[:k]
        old_medoids = np.array([-1]*k)
        new_medoids = np.array([-1]*k)

        # Until the medoids stop updating, do the following:
        while not ((old_medoids == curr_medoids).all()):
            nearest_medoid_idx = self.assign_points_to_clusters(curr_medoids, dist_mat)
            # Update cluster medoids to be lowest cost point. 
            for curr_medoid in curr_medoids:
                
                curr_cluster = np.where(nearest_medoid_idx == curr_medoid)[0]
                #Compute the distances inside the cluster, and pickup the most suitable data point as the medoid 
                new_medoids[curr_medoids == curr_medoid] = self.__compute_new_medoid(curr_cluster, dist_mat)
            
            old_medoids[:] = curr_medoids[:]
            curr_medoids[:] = new_medoids[:]

                
        return curr_medoids, nearest_medoid_idx

    @staticmethod
    def assign_points_to_clusters(medoids, dist_mat):
        """
        Assign each point to cluster with closest medoid.
        Args
        - medoids: A list of indices in the distance matrix as medoids
        - dist_mat: distance matrix.
        Output
        - nearest_medoid_idx: A list with the size of data points in the distance matrix. The index is the data point index as in the distance matrix.
        The value is the index of the nearest medoid.
        """
        distances_to_medoids = dist_mat[:, medoids]
        nearest_medoid_idx = medoids[np.argmin(distances_to_medoids, axis=1)]
        nearest_medoid_idx[medoids] = medoids
        return nearest_medoid_idx

    @staticmethod
    def __compute_new_medoid(cluster, dist_mat):
        """
        Minimize the total distances to medoids. Maskind the distance matrix to compute the total distances to the nearest medoid.
        Args
        - cluster: list of indices of members of a cluster
    - dist_mat: distance matrix.
        Output
        - costs: The total distances of all cluster members to their nearest medoid.
        """
        mask = np.ones(dist_mat.shape)
        mask[np.ix_(cluster, cluster)] = 0 #Grid for all the medoids rows and column
        cluster_distances = np.ma.masked_array(data=dist_mat, mask=mask, fill_value=10e9) #Mask all the values not relevant to the medoid.
        costs = cluster_distances.sum(axis=1)
        return costs.argmin(axis=0, fill_value=10e9)


