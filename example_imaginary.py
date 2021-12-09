import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from active_learning.core import ActiveLearner, MAL1, MismatchFirstFarthestTraversal, LargestNeighborhood


def target_func(X):
    y = np.zeros(len(X))
    for i in range(len(y)):
        if (X[i,0]> 0.2 and X[i,1]>0.2) and (X[i,1]<0.7-X[i,0]):
            y[i] = 1
        elif (X[i,0]< 0.8 and X[i,1]<0.8) and (X[i,1]>1.3-X[i,0]):
            y[i] = 1        
        else:
            y[i] = 0
    return y

def data_generation(n=5000):
    np.random.seed(0)
    X = np.random.random([n, 2])
    y = target_func(X)
    return X, y

def plot_data(X, y):
    ax = plt.axes()
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    
    plt.plot([0.2,0.5], [0.2,0.2], 'r-')
    plt.plot([0.2,0.2], [0.2,0.5], 'r-')
    plt.plot([0.2,0.5], [0.5,0.2], 'r-')
    
    plt.plot([0.8,0.5], [0.8,0.8], 'r-')
    plt.plot([0.8,0.8], [0.5,0.8], 'r-')
    plt.plot([0.8,0.5], [0.5,0.8], 'r-')

    for i in range(len(X)):
        if y[i] == 1: 
            plt.plot(X[i,0], X[i,1], 'r.')
        else:
            plt.plot(X[i,0], X[i,1], 'g.')

    plt.show()


def random_sampling_process(X_train, y_train, X_test, y_test):
    "The decision boundary cannot be learned LogisticRegression. Decision tree is used here."
    learner = ActiveLearner(X_train,
                            initial_batch_size=100,
                            batch_size=100,
                            classifier=GradientBoostingClassifier())
    
    # n_batch = int(len(X_train) / learner.batch_size)
    n_batch = 5
    print("Query strategy: Mismatch-first farthest-traversal...")
    for i in range(n_batch):
        print("Batch {0}:".format(i + 1))
        batch = learner.draw_next_batch()
        learner.annotate_batch(batch, y_train[batch])
        print("Annotated instances: {0}".format(len(learner.L)))

        
        plot_data(X_train[learner.L], y_train[learner.L])
        
        print("Training starts.")
        learner.train()
        print("Training is done.")
        
        y_test_pred = learner.classifier.predict(X_test)
        
        f1 = metrics.f1_score(y_test, y_test_pred, average='macro')
        print("The average F1 score is ", f1)


def mismatch_first_farthest_traversal(X_train, y_train, X_test, y_test):
    "The decision boundary cannot be learned LogisticRegression. Decision tree is used here."
    learner = MismatchFirstFarthestTraversal(X_train,
                                             initial_batch_size=100,
                                             batch_size=100,
                                             classifier=GradientBoostingClassifier())
    learner.dist_metric = 'euclidean'
    learner.medoids_sort = 'distance'
    # n_batch = int(len(X_train) / learner.batch_size)
    n_batch = 5
    print("Query strategy: Mismatch-first farthest-traversal...")
    for i in range(n_batch):
        print("Batch {0}:".format(i + 1))
        batch = learner.draw_next_batch()
        learner.annotate_batch(batch, y_train[batch])
        print("Annotated instances: {0}".format(len(learner.L)))

        
        #plot_data(X_train[learner.L], y_train[learner.L])
        
        print("Training starts.")
        learner.train()
        print("Training is done.")
        
        y_test_pred = learner.classifier.predict(X_test)
        plot_data(X_test, y_test_pred)

        f1 = metrics.f1_score(y_test, y_test_pred, average='macro')
        print("The average F1 score is ", f1)









#uncertainty_sampling(X)
if __name__ == '__main__':
    X_train, y_train = data_generation(n=5000)
    X_test, y_test = data_generation(n=1000)
    #random_sampling_process(X_train, y_train, X_test, y_test)
    plot_data(X_train, y_train)
    mismatch_first_farthest_traversal(X_train, y_train, X_test, y_test)
#MFFT(X)

