import numpy as np 
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time


def compute_cosine_dist_mat(X):
    "The computation of cosine distance matrix from vectorized instances"

    dot_product = np.dot(X, X.T)
    vec_norms = np.linalg.norm(X, axis=1)
    norm_product = \
        np.outer(vec_norms, vec_norms)
    return dot_product / norm_product


class MismatchFirstFarthestTraversal:
    "The implementation of mismatch-first farthest-traversal"

    def __init__(self, X, batch_size=20, initial_batch_size=400, classifier=None):
        if classifier is None:
            self.classifier = LogisticRegression()
        else:
            self.classifier = classifier

        self.batch_size = batch_size
        self.initial_batch_size = initial_batch_size
        self.n_batch = 0  # Starting the first batch

        # Labeled set initially empty
        self.L = np.array([])
        self.U = np.arange(X.shape[0])

        # Distance matrix, the vectors are assumed to be normalized (Norm of embedding vector is 1).
        # Thus, inner product is equivalent to cosine similarity
        self.dist_mat = np.dot(X, X.T)
        self.dist_mat = 1 - self.dist_mat.toarray()

    def farthest_traversal(self, n, U=None):
        """
        Farthest-first traversal n samples on the whole unlabeled set, or subset without matched prediction
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
        print (len(set(selection_batch)))
        return np.array(selection_batch)

    def draw_next_batch(self):
        if self.n_batch == 0:
            return self.farthest_traversal(self.initial_batch_size)


if __name__ == '__main__':
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    vectorizer = TfidfVectorizer()
    classifier = LogisticRegression()
    
    # The vectors are normalized, thus inner product is equivalent to cosine similarity
    X_train = vectorizer.fit_transform(newsgroups_train.data)
    y_train = newsgroups_train.target


    learner = MismatchFirstFarthestTraversal(X_train)
    learner.initial_batch_size = 1000

    """
    # Testing if nearest neighbour classification works
    for i in range(100):
        learner.dist_mat[i, i] = 1  # Exclued itself for nn
        nn = np.argmin(learner.dist_mat[i])
        print (y_train[i], y_train[nn])
    """



    # selection = learner.draw_next_batch()
    
    # selection = np.random.randint(0, X_train.shape[0], 1000)
    # print(y_train[selection])

    """
    print("Training starts.")
    classifier.fit(X_train[selection], y_train[selection])
    print("Training is done.")
    
    X_test = vectorizer.transform(newsgroups_test.data)
    y_test = newsgroups_test.target
    y_test_pred = classifier.predict(X_test)

    f1 = metrics.f1_score(y_test, y_test_pred, average='macro')
    print("The average F1 score is ", f1)
    """

    
