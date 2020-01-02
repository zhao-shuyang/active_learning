import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import sklearn.metrics.pairwise
from sklearn.ensemble import GradientBoostingClassifier


np.random.seed(0)

n = 5000
batch_size = 100
X = np.random.random([n,2])
#cla = svm.SVC(kernel='rbf', gamma='auto', probability=True, tol=1e-8, class_weight='balanced')
cla = GradientBoostingClassifier()
#cla = svm.SVC(kernel='linear')
#cla2 = svm.SVC(kernel='linear')

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
    
def random_sampling(X, batch_size):
    selected_indices = np.random.permutation(len(X))
    return (selected_indices[:batch_size])

def plot_data(X, y):
    ax = plt.axes()
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    plt.plot([0.2,0.5], [0.2,0.2], 'b-')
    plt.plot([0.2,0.2], [0.2,0.5], 'b-')
    plt.plot([0.2,0.5], [0.5,0.2], 'b-')
    
    plt.plot([0.8,0.5], [0.8,0.8], 'b-')
    plt.plot([0.8,0.8], [0.5,0.8], 'b-')
    plt.plot([0.8,0.5], [0.5,0.8], 'b-')
    

    for i in range(len(X)):
        if y[i] == 1: 
            plt.plot(X[i,0], X[i,1], 'b.')
        else:
            plt.plot(X[i,0], X[i,1], 'g.')
    #ax.set_title("Model")
    plt.show()


def plot_data2(X, y, L, S):
    ax1 = plt.axes()
    ax1.set_xlim([0,1])
    ax1.set_ylim([0,1])

    for i in range(len(L)):
        if y[L][i] == 1: 
            ax1.plot(X[L][i,0], X[L][i,1], 'b.')
        else:
            ax1.plot(X[L][i,0], X[L][i,1], 'g.')

    for i in range(len(S)):
        if y[S][i] == 1: 
            ax1.plot(X[S][i,0], X[S][i,1], 'bo')
        else:
            ax1.plot(X[S][i,0], X[S][i,1], 'go')
            
    ax1.set_title("Labeling budget {0}".format(len(L)))
            
    plt.show()
    

def uncertainty_sampling(X):
    L = np.array([])
    U = np.arange(len(X))
    #plot_data(X, target_func(X))
    y =  target_func(X)
    while len(L) < len(X):
        print (len(L), len(X))
        if not L.any():
            L = random_sampling(U, batch_size)
            plot_data2(X, y, L, L)
            
        elif len(set(target_func(X[L]).tolist())) == 1:
            S = random_sampling(U, batch_size)
            L = np.concatenate((L,S))
            U = np.array(list(set(U.tolist()) - set(S.tolist())))
            plot_data2(X, y, L, S)
        else:
            cla.fit(X[L], target_func(X[L]))
            plot_data(X, cla.predict(X))

            #S = random_sampling(U, batch_size)
            
            certainty = 2*np.abs(0.5 - cla.predict_proba(X[U])[:,0])
            order = np.argsort(certainty)
            S = U[order[:batch_size]]
            L = np.concatenate((L,S))
            plot_data2(X, y, L ,S)            
            U = np.array(list(set(U.tolist()) - set(S.tolist())))
            #print (cla.predict(X))
           

        #plot_data(X[L], target_func(X[L]))
       

def farthest_traversal(X):
    y =  target_func(X)
    
    L = np.array([np.random.randint(len(X))])
    U = np.arange(len(X))
    U = np.array(list(set(U.tolist()) - set(L.tolist())))
    S = np.array([])
    plot_data(X, target_func(X))
    dist_mat = sklearn.metrics.pairwise.euclidean_distances(X)
    print (dist_mat.shape)
    
    n_annotated = 1
    while len(L) < len(X):
        dist = np.zeros(len(U))

        for i in range(len(U)):
            dist[i] = np.min(dist_mat[U[i]][L])

        s = U[np.argmax(dist)]


        if S.any():                    
            S = np.array(S.tolist() +[s])
        else:
            S = np.array([s])
            
        L = np.array(L.tolist() +[s])
        U = np.array(list(set(U.tolist()) - set(S.tolist())))
        
        n_annotated += 1
        if n_annotated%batch_size == 0:
            plot_data2(X, y, L, S)
            S = np.array([])
            cla.fit(X[L], target_func(X[L]))
            plot_data(X, cla.predict(X))
            
            


def MFFT(X):
    L = np.array([np.random.randint(len(X))])
    U = np.arange(len(X))
    U = np.array(list(set(U.tolist()) - set(L.tolist())))

    plot_data(X, target_func(X))
    dist_mat = sklearn.metrics.pairwise.euclidean_distances(X)
    print (dist_mat.shape)

    n_batch = 1
    while len(L) < len(X):
        if n_batch == 1:
            for i in range(batch_size -1):
                dist = np.zeros(len(U))
                for j in range(len(U)):
                    dist[j] = np.min(dist_mat[U[j]][L])
                s = U[np.argmax(dist)]                
                
                L = np.array(L.tolist()+[s])

                U = np.array(list(set(U.tolist()) - set(L.tolist())))
            
            plot_data2(X, target_func(X), L, L)
            n_batch += 1
            
        else:
            cla.fit(X[L], target_func(X[L]))
            y1all = cla.predict(X)
            y1 = y1all[U]
            plot_data(X, y1all)
            
            y2 = nnp(X, L, U)
            M1 = U[np.where(np.abs(y1 - y2)==1)]
            M2 = np.array(list(set(U.tolist()) - set(M1.tolist())))
            
            S = np.array([])                
            for i in range(batch_size):
                print (len(L), len(M1), len(M2))
                if len(M1) > 0:
                    dist = np.zeros(len(M1))
                    for j in range(len(M1)):
                        dist[j] = np.min(dist_mat[M1[j]][L])
                    s = M1[np.argmax(dist)]
                    M1 = np.array(list(set(M1.tolist()) - set(S.tolist())))
                else:
                    dist = np.zeros(len(M2))
                    for j in range(len(M2)):
                        dist[j] = np.min(dist_mat[M2[j]][L])
                    s = M2[np.argmax(dist)]
                    M2 = np.array(list(set(M2.tolist()) - set(S.tolist())))


                if S.any():                    
                    S = np.array(S.tolist() +[s])
                else:
                    S = np.array([s])
                    
                L = np.array(L.tolist() +[s])

                U = np.array(list(set(U.tolist()) - set(S.tolist())))

            plot_data2(X, target_func(X), L, S)

def nnp(X, L, U):
    dist_mat = sklearn.metrics.pairwise.euclidean_distances(X)
    y = target_func(X)
    y2 = np.zeros(len(U))
    for i in range(len(U)):        
        nearest_neighbour = L[np.argmin(dist_mat[U[i]][L])]
        print (nearest_neighbour)
        y2[i] = y[nearest_neighbour]
    return y2
    
#uncertainty_sampling(X)
farthest_traversal(X)
#MFFT(X)

