from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from active_learning.core import MismatchFirstFarthestTraversal, MismatchFirstLargestNeighborhood

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
    
    n_batch = 35
    #learner = MismatchFirstFarthestTraversal(X_train, initial_batch_size=500, batch_size=100)
    learner = MismatchFirstLargestNeighborhood(X_train, initial_batch_size=500, batch_size=100)
    
    for i in range(n_batch):
        print("Batch {0}:".format(i + 1))
        batch = learner.draw_next_batch()
        learner.annotate_batch(batch, y_train[batch])
        print("Annotated instances: {0}".format(len(learner.L)))
        
        print("Training starts.")
        # learner.train()
        learner.train_with_propagated_labels()

        print("Training is done.")
        y_test_pred = learner.classifier.predict(X_test)

        f1 = metrics.f1_score(y_test, y_test_pred, average='macro')
        print("The average F1 score is ", f1)
