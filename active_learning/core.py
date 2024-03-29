import collections
import numpy as np
from sklearn.linear_model import LogisticRegression
import sklearn.metrics.pairwise
from . import cluster


class ActiveLearner:
    "The general framework of batch-mode pool-based active learning algorithm."
    def __init__(self, X, batch_size=20, initial_batch_size=None, classifier=LogisticRegression(max_iter=200)):
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

    def compute_dist_mat(self, X, metric='cosine'):
        """
        The computation of distance matrix based on inner product.
        It is equivalent to cosine distance when vector representations are normalized.
        """
        # dist_mat = np.dot(X, X.T)
        if metric == 'cosine':
            return self.compute_cosine_dist_mat(X)
        dist_mat = sklearn.metrics.pairwise.euclidean_distances(X)
        print("Computation of distance matrix finished.")
        return dist_mat

    @staticmethod
    def compute_cosine_dist_mat(X):
        "The computation of cosine distance matrix from vectorized instances"
        dot_product = np.dot(X, X.T)
        vec_norms = np.linalg.norm(X, axis=1)
        norm_product = np.outer(vec_norms, vec_norms)
        return 1 - dot_product / norm_product

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

            for i in range(batch_size):
                if self.medoids:
                    selection_batch.append(self.medoids.popleft())
                else:
                    break
                    print("Medoids are exhausted.")
            self.n_batch += 1
            return np.array(selection_batch)

    def train_with_propagated_labels(self):
        tmp_P = [] 
        for sample in self.L:
            for cluster_member in self.cluster_analyzer.clusters[sample]:
                self.y[cluster_member] = self.y[sample]
                tmp_P.append(cluster_member)
        self.P = np.array(tmp_P)

        self.classifier.fit(self.X[self.P], self.y[self.P])



class MismatchFirstFarthestTraversal(MAL1):
    """
    The implementation of mismatch-first farthest-traversal.
    It is supposed to be faster since it does not require the slow PAM algorithm.
    This is used in case that some classes are rare.
    """

    def __init__(self, *args, **kwargs):
        super(MismatchFirstFarthestTraversal, self).__init__(*args, **kwargs)
        self.dist_metric = 'cosine'
        self.medoids_sort = 'size'
        
    def draw_next_batch(self):
        "Generate medoids and draw the medoids from a deque."

        if not hasattr(self, 'dist_mat'):
            self.dist_mat = self.compute_dist_mat(self.X, self.dist_metric)
            
        if not hasattr(self, 'cluster_analyzer'):
            self.cluster_analyzer = cluster.KMedoidClustering(self.dist_mat, self.K)
            # self.cluster_analyzer = self.clustering(self.dist_mat, self.K)
            self.cluster_analyzer.medoids = cluster.farthest_search(self.dist_mat, self.K)
            self.cluster_analyzer.repartition()
            
            if self.medoids_sort == 'size':
                self.cluster_analyzer.sort_clusters(order_by='size')
            self.medoids = collections.deque(self.cluster_analyzer.medoids)
            print (self.medoids)
            

        selection_batch = []

        while len(selection_batch) < self.batch_size:
            # Farthest traversal for the frist batch
            if self.medoids:
                selection_batch.append(self.medoids.popleft())
            else:
                selection_size = self.batch_size - len(selection_batch)
                matched_mask, mismatched_mask = self.compare_predictions()
                
                if selection_size <= np.sum(mismatched_mask):
                    selection = self.farthest_traversal(selection_size, np.nonzero(mismatched_mask)[0])                     
                else:
                    selectionA = np.nonzero(mismatched_mask)[0]
                    selectionB = self.farthest_traversal(selection_size - np.sum(mismatched_mask), np.nonzero(matched_mask)[0])
                    selection = np.concatenate((selectionA, selectionB))
                    print (selectionA, selectionB)
                
                selection_batch += selection.tolist()
                
        self.n_batch += 1
        return np.array(selection_batch)
                                    
    def farthest_traversal(self, n, U=None):
        """        
        Farthest-first traversal on a subset with respect to labeled data points.
        """
        selection_samples = []

        nn_dist = np.zeros(len(U))
        for i in range(len(U)):
            nn_dist[i] = np.min(self.dist_mat[U[i]][self.L])
    
        for i in range(n):
            if not self.L.any():  # The very first sample is randomly selected
                s_index = np.random.randint(len(U))
            else:
                s_index = np.argmax(nn_dist)
                update_mask = self.dist_mat[U[s_index]][U] < nn_dist
                nn_dist[update_mask] = self.dist_mat[U[s_index]][U[update_mask]]
            selection_samples.append(U[s_index])

        return np.array(selection_samples)

    def compare_predictions(self):
        """
        Make model predictons and label propagations
        """

        self.train()
        y_pred = self.classifier.predict(self.X)
        
        y_prop = np.zeros(y_pred.shape)
        self.cluster_analyzer.medoids = self.L  # Use labeled data points as medoids
        self.cluster_analyzer.repartition()  # Update the clusters
        for sample in self.cluster_analyzer.medoids:
            for cluster_member in self.cluster_analyzer.clusters[sample]:
                y_prop[cluster_member] = self.y[sample]
                
        mismatched_mask = (y_pred != y_prop).astype(int)
        unlabeled_mask = np.zeros(self.X.shape[0], dtype=int)
        unlabeled_mask[self.U] = 1
        matched_mask = unlabeled_mask - mismatched_mask
        print("Among {0} unlabeled samples, those with \n MISmatched predictions: {1} \n matched predictions: {2}".format(np.sum(unlabeled_mask), np.sum(mismatched_mask), np.sum(matched_mask)))
        return matched_mask, mismatched_mask

    def train_with_propagated_labels(self):
        tmp_P = []
            
        if len(self.L) > self.K:
            self.cluster_analyzer.medoids = self.L
            self.cluster_analyzer.repartition()
            
        for sample in self.L:
            for cluster_member in self.cluster_analyzer.clusters[sample]:
                self.y[cluster_member] = self.y[sample]
                tmp_P.append(cluster_member)
        self.P = np.array(tmp_P)
        self.classifier.fit(self.X[self.P], self.y[self.P])


class LargestNeighborhood(MismatchFirstFarthestTraversal):
    """
    This is supposed to be used in cases class distribution is even.
    """
    def draw_next_batch(self):
        "Generate medoids and draw the medoids from a deque."

        if not hasattr(self, 'dist_mat'):
            self.dist_mat = self.compute_dist_mat(self.X)

        if not hasattr(self, 'cluster_analyzer'):
            self.cluster_analyzer = cluster.KMedoidClustering(self.dist_mat, self.K)
            traversal_size = int((self.initial_batch_size * self.X.shape[0])**0.5)
            print ("traversal_size: {0}. Among them, the {1} samples with largest neighborhood will be selected.".format(traversal_size, self.initial_batch_size))
                    
            self.cluster_analyzer.medoids = cluster.farthest_search(self.dist_mat, traversal_size)
            self.cluster_analyzer.repartition()
            self.cluster_analyzer.sort_clusters()
            self.medoids = collections.deque(self.cluster_analyzer.medoids[:self.initial_batch_size])

        selection_batch = []
            
        while len(selection_batch) < self.batch_size:
            # Farthest traversal for the frist batch
            if self.medoids:
                selection_batch.append(self.medoids.popleft())
            else:
                selection_size = self.batch_size - len(selection_batch)
                traversal_size = int((selection_size * len(self.U))**0.5)
                print ("traversal_size: {0}. Among them, the {1} samples with largest neighborhood will be selected.".format(traversal_size, self.batch_size))
                traversal_pool = self.farthest_traversal(traversal_size, self.U)
                selection = self.largest_neighborhood(selection_size, traversal_pool)
                
                # matched_mask, mismatched_mask = self.compare_predictions()
                """
                if selection_size <= np.sum(mismatched_mask):
                    # traversal_size = int((selection_size * np.sum(mismatched_mask))**0.5)
                    traversal_size = int((selection_size * len(self.U))**0.5)
                    
                    print ("traversal_size: {0}".format(traversal_size))
                    traversal_pool = self.farthest_traversal(traversal_size, self.U)
                    selection = self.largest_neighborhood(selection_size, traversal_pool)
                else:
                    selectionA = np.nonzero(mismatched_mask)[0]
                    selectionB = self.largest_neighborhood(selection_size - np.sum(mismatched_mask), np.nonzero(matched_mask)[0])
                    selection = np.concatenate((selectionA, selectionB))
                """
                selection_batch += selection.tolist()

        self.n_batch += 1
        return np.array(selection_batch)


    def largest_neighborhood(self, n, U=None):
        """        
        Selecting samples with largest neighborhood of -first traversal on a subset with respect to labeled data points.
        """
        self.cluster_analyzer.medoids = np.concatenate((self.L, U))
        # This U means unlabeled data with mismatched predictions if exists, otherwise labeled
        
        self.cluster_analyzer.repartition()
        self.cluster_analyzer.sort_clusters(order_by='size')
        
        selection_samples = []

        for medoid in self.cluster_analyzer.medoids:
            if len(selection_samples) >= n:
                return np.array(selection_samples)

            elif medoid not in self.L:
                selection_samples.append(medoid)
                
        return np.array(selection_samples)

                
