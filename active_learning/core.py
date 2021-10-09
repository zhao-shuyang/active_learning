import collections
import numpy as np
from sklearn.linear_model import LogisticRegression

from . import cluster


class ActiveLearner:
    "The general framework of batch-mode pool-based active learning algorithm."
    def __init__(self, X, batch_size=20, initial_batch_size=None, classifier=LogisticRegression()):
        "The default classifier to be used is logistic regression"
        
        self.batch_size = batch_size
        self.initial_batch_size = (
            initial_batch_size if initial_batch_size is not None
            else batch_size)
        self.n_batch = 0  # Starting the first batch

        self.X = X  # Training set, each row is a vectorized instance
        self.y = np.zeros(X.shape[0])  # The labels, assuming initially unlabeled
        self.L = np.array([])  # indices of labeled instances
        self.U = np.arange(X.shape[0])  # indices of unlabeled instances

        self.classifier = classifier
        
    def draw_next_batch(self):
        "The query strategy. This will be overriden by different active learning methods."
        s_indices = np.random.randint(0, len(self.U), self.batch_size)
        self.n_batch += 1
        return self.U[s_indices]

    def annotate_batch(self, selection_batch, batch_target):
        self.y[selection_batch] = batch_target
        self.L = np.array(self.L.tolist() + selection_batch.tolist())  # Labeled and to be labeled in the same batch
        self.U = np.array(list(set(self.U.tolist()) - set(selection_batch.tolist())))
        
    def train(self):
        self.classifier.fit(self.X[self.L], self.y[self.L])


class MAL1(ActiveLearner):
    "This is a simple medoid-based active learner. Clustering is performed only once here"
    def __init__(self, *args, **kwargs):
        super(MAL1, self).__init__(*args, **kwargs)
        self.P = np.array([])  # indices of instances with propagated labels
        self.K = int(self.X.shape[0] / 4)  # Average cluster size of 4

    @staticmethod
    def compute_dist_mat(X):
        """
        The computation of distance matrix based on inner product.
        It is equivalent to cosine distance when vector representations are normalized.
        """
        dist_mat = np.dot(X, X.T)
        dist_mat = 1 - dist_mat.toarray()
        print("Computation of distance matrix finished.")
        return dist_mat

    @staticmethod
    def compute_cosine_dist_mat(X):
        "The computation of cosine distance matrix from vectorized instances"
        dot_product = np.dot(X, X.T)
        vec_norms = np.linalg.norm(X, axis=1)
        norm_product = np.outer(vec_norms, vec_norms)
        return dot_product / norm_product

    @staticmethod
    def clustering(dist_mat, K, order_by="size"):
        cluster_analyzer = cluster.KMedoidClustering(dist_mat, K)
        cluster_analyzer.perform_clustering()
        cluster_analyzer.sort_clusters(order_by=order_by)
        return cluster_analyzer

    def draw_next_batch(self):
        "Generate medoids and draw the medoids from a deque."
        if not hasattr(self, 'dist_mat'):
            self.dist_mat = self.compute_dist_mat(self.X)
        if not hasattr(self, 'cluster_analyzer'):
            self.cluster_analyzer = self.clustering(self.dist_mat, self.K)
            self.medoids = collections.deque(self.cluster_analyzer.medoids)
        selection_batch = []
        if hasattr(self, 'medoids'):
            batch_size = (self.batch_size if self.n_batch == 0
                          else self.initial_batch_size)
            print (batch_size)
            for i in range(batch_size):
                if self.medoids:
                    selection_batch.append(self.medoids.popleft())
                else:
                    break
                    print("Medoids are exhausted.")
            self.n_batch += 1
            return np.array(selection_batch)
                   
    def annotate_batch(self, selection_batch, batch_target):
        "In addition to annotation, propagate the labels from medoids."
        super(MAL1, self).annotate_batch(selection_batch, batch_target)
        
        
        # self.y[selection_batch] = batch_target
        # self.L = np.array(self.L.tolist() + selection_batch.tolist())  # Labeled and to be labeled in the same batch
        # self.U = np.array(list(set(self.U.tolist()) - set(selection_batch.tolist())))

        tmp_P = self.P.tolist()
        for i, sample in enumerate(selection_batch):
            self.y[sample] = batch_target[i]
            for cluster_member in self.cluster_analyzer.clusters[sample]:
                self.y[cluster_member] = batch_target[i]
                tmp_P.append(cluster_member)
        self.P = np.array(tmp_P)
        
    def train_with_propagated_labels(self):
        self.classifier.fit(self.X[self.P], self.y[self.P])


class MismatchFirstFarthestTraversal(MAL1):
    """
    The implementation of mismatch-first farthest-traversal.
    It is supposed to be faster since it does not require the slow PAM algorithm.
    """

    def __init__(self, *args, **kwargs):
        super(MismatchFirstFarthestTraversal, self).__init__(*args, **kwargs)
        self.K = int(self.X.shape[0] / 10)  # Average cluster size of 10. The initial batch size should be smaller than 10% of the data.

        
    def draw_next_batch(self):
        "Generate medoids and draw the medoids from a deque."
        batch_size = (self.batch_size if self.n_batch == 0
                      else self.initial_batch_size)

        if not hasattr(self, 'dist_mat'):
            self.dist_mat = self.compute_dist_mat(self.X)

        if not hasattr(self, 'cluster_analyzer'):
            self.cluster_analyzer = cluster.KMedoidClustering(self.dist_mat, self.K)
            self.cluster_analyzer.medoids = cluster.farthest_search(self.dist_mat, self.K)  # Acutually only the two lines of code is different
            print (self.cluster_analyzer.medoids)
            self.cluster_analyzer.repartition()
            self.cluster_analyzer.sort_clusters()
            print (self.cluster_analyzer.medoids)
            
            self.medoids = collections.deque(self.cluster_analyzer.medoids)

        selection_batch = []
        
        if hasattr(self, 'medoids'):
            batch_size = (self.batch_size if self.n_batch == 0
                          else self.initial_batch_size)
            print (batch_size)
            for i in range(batch_size):
                if self.medoids:
                    selection_batch.append(self.medoids.popleft())
                else:
                    break
                    print("Medoids are exhausted.")
            self.n_batch += 1
            return np.array(selection_batch)

    
    def farthest_traversal(self, n, U=None):

        """        
        Farthest-first traversal on a subset, assumed with mismatched predictons.
        """
        tmp_U = np.copy(self.U) if U is None else np.copy(U)  # Unlabeled data, without to be labeled in the batch
        tmp_L = np.copy(self.L)  # Labeled data and to be labeled in the batch
        selection_batch = []

        for i in range(n):
            if not tmp_L.any():  # The very first sample is randomly selected
                s_index = np.random.randint(len(tmp_U))
            else:
                nn_dist = np.zeros(len(tmp_U))  # Store the nearest neighbour distance
                for j in range(len(tmp_U)):
                    nn_dist[j] = np.min(self.dist_mat[tmp_U[j]][tmp_L])
                s_index = np.argmax(nn_dist)

            s = tmp_U[s_index]
            selection_batch.append(s)
            tmp_L = np.array(self.L.tolist()+[selection_batch])  # Labeled and to be labeled in the same batch
            tmp_U = np.delete(tmp_U, s_index)

        return np.array(selection_batch)
