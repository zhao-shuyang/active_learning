from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from active_learning.core import ActiveLearner, MismatchFirstFarthestTraversal, LargestNeighborhood


def fully_supervised(X_train, y_train, X_test, y_test):
    ceiling_classifier = LogisticRegression()
    ceiling_classifier.fit(X_train, y_train)
    y_test_pred = ceiling_classifier.predict(X_test)
    f1 = metrics.f1_score(y_test, y_test_pred, average='macro')
    print ("With all the training dataset labeled, \
    the classification accuracy is  {0}.".format(f1))


def random_sampling(X_train, y_train, X_test, y_test):
    learner = ActiveLearner(X_train, initial_batch_size=200, batch_size=200)
    n_batch = 50
    print("Query strategy: Random sampling...")
    for i in range(n_batch):
        print("Batch {0}:".format(i + 1))
        batch = learner.draw_next_batch()
        learner.annotate_batch(batch, y_train[batch])
        print("Annotated instances: {0}".format(len(learner.L)))
        
        print("Training starts.")
        learner.train()
        print("Training is done.")
        
        y_test_pred = learner.classifier.predict(X_test)

        f1 = metrics.f1_score(y_test, y_test_pred, average='macro')
        print("The average F1 score is ", f1)


def largest_neighborhood(X_train, y_train, X_test, y_test):
    learner = LargestNeighborhood(X_train, batch_size=200, initial_batch_size=1000)
    n_batch = 50

    print("Query strategy: Mismatch first largest neigbourhood...")

    for i in range(n_batch):
        print("Batch {0}:".format(i + 1))
        batch = learner.draw_next_batch()
        learner.annotate_batch(batch, y_train[batch])
        print("Annotated instances: {0}".format(len(learner.L)))
        
        print("Training starts.")
        learner.train_with_propagated_labels()
        print("Training is done.")
        
        y_test_pred = learner.classifier.predict(X_test)

        f1 = metrics.f1_score(y_test, y_test_pred, average='macro')
        print("The average F1 score is ", f1)

def mismatch_first_farthest_traversal(X_train, y_train, X_test, y_test):
    learner = MismatchFirstFarthestTraversal(X_train, initial_batch_size=2000, batch_size=200)
    learner.K = 2000
    n_batch = 50
    print("Query strategy: Mismatch-first farthest-traversal...")
    for i in range(n_batch):
        print("Batch {0}:".format(i + 1))
        if learner.n_batch == 0:
            learner.batch_size = 2000
        else:
            learner.batch_size = 200
        
        batch = learner.draw_next_batch()
        learner.annotate_batch(batch, y_train[batch])
        print("Annotated instances: {0}".format(len(learner.L)))
        
        print("Training starts.")
        learner.train_with_propagated_labels()
        print("Training is done.")
        
        y_test_pred = learner.classifier.predict(X_test)

        f1 = metrics.f1_score(y_test, y_test_pred, average='macro')
        print("The average F1 score is ", f1)


if __name__ == '__main__':
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    vectorizer = TfidfVectorizer()
    classifier = LogisticRegression()
    
    # The vectors are normalized, thus inner product is equivalent to cosine similarity
    X_train = vectorizer.fit_transform(newsgroups_train.data)
    y_train = newsgroups_train.target
    X_test = vectorizer.transform(newsgroups_test.data)
    y_test = newsgroups_test.target

    # fully_supervised(X_train, y_train, X_test, y_test)
    # The accuracy with all the data labeled should be aroung 81%-82%
    

    random_sampling(X_train, y_train, X_test, y_test)
    mismatch_first_largest_neighborhood(X_train, y_train, X_test, y_test)
    mismatch_first_farthest_traversal(X_train, y_train, X_test, y_test)
