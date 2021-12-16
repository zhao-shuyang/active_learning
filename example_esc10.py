import os
import torch
import soundfile
import librosa
import pandas as pd
import numpy as np
import h5py

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from active_learning.core import ActiveLearner, MAL1, MismatchFirstFarthestTraversal, LargestNeighborhood
from active_learning.net_arch import Cnn14


model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,fmin=50,fmax=14000,classes_num=527)
checkpoint = torch.load('weights/Cnn14_mAP=0.431.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'])

model.eval()


def get_features(audio_file):
    sig, sr = soundfile.read(audio_file)
    
    if len(sig.shape) > 1:
        sig = sig[:, 0]

    mel = librosa.feature.melspectrogram(sig, sr=32000, win_length=1024, hop_length=320, fmin=50, fmax=14000, n_mels=64)
    
    with torch.no_grad():
        mel = librosa.feature.melspectrogram(sig, sr=32000, win_length=1024, hop_length=320, fmin=50, fmax=14000, n_mels=64)

        emb = model(torch.from_numpy(mel.T).unsqueeze(0).unsqueeze(0).float())

    return emb


def generate_dataset(audio_root, file_meta, test_fold=1):
    df_esc50 = pd.read_csv(file_meta, sep=',')
    class_list = sorted(df_esc50.loc[df_esc50['esc10']==True]['category'].unique())
    
    df_esc10_train = df_esc50[(df_esc50['fold'] != test_fold) & (df_esc50['esc10']==True)]
    df_esc10_test = df_esc50[(df_esc50['fold'] == test_fold) & (df_esc50['esc10']==True)]
    emb_size = 2048
    train_size = len(df_esc10_train)
    test_size = len(df_esc10_test)

    X_train = np.zeros((train_size, emb_size))
    y_train = np.zeros(train_size)
    
    X_test = np.zeros((test_size, emb_size))
    y_test = np.zeros(test_size)
    
    for index, (_, row) in enumerate(df_esc10_train.iterrows()):
        audio_file_path = os.path.join(audio_root, row['filename'])
        label = class_list.index(row['category'])
        print (audio_file_path, label)
        feature = get_features(audio_file_path)[0]
        X_train[index] = feature
        y_train[index] = label

    for index, (_, row) in enumerate(df_esc10_test.iterrows()):
        audio_file_path = os.path.join(audio_root, row['filename'])
        label = class_list.index(row['category'])
        print (audio_file_path, label)
        feature = get_features(audio_file_path)[0]
        X_test[index] = feature
        y_test[index] = label
    
    return X_train, y_train, X_test, y_test


def save_datasets():
    for i_fold in range(5):
        print (f'Fold {i_fold+1}:')
        f = h5py.File(f'esc10_fold_{i_fold+1}.hdf5', 'w')
        X_train, y_train, X_test, y_test = generate_dataset("ESC-50/audio", "ESC-50/meta/esc50.csv", i_fold+1)
        f.create_dataset("X_train", data=X_train)
        f.create_dataset("X_test", data=X_test)
        f.create_dataset("y_train", data=y_train)
        f.create_dataset("y_test", data=y_test)
        f.close()


def random_sampling(h5_path):
    f = h5py.File(h5_path, 'r')
    X_train, y_train, X_test, y_test = f["X_train"][:], f["y_train"][:], f["X_test"][:], f["y_test"][:]  
    learner = ActiveLearner(X_train, initial_batch_size=20, batch_size=20, classifier=LogisticRegression(max_iter=500))
    n_batch = 16
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


def mismatch_first_farthest_traversal(h5_path):
    f = h5py.File(h5_path, 'r')
    X_train, y_train, X_test, y_test = f["X_train"][:], f["y_train"][:], f["X_test"][:], f["y_test"][:]

    learner = MismatchFirstFarthestTraversal(X_train, initial_batch_size=20, batch_size=20, classifier=LogisticRegression(max_iter=500))
    learner.K = 80
    n_batch = 16
    print("Query strategy: Mismatch-first farthest-traversal...")
    for i in range(n_batch):
        
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
    # random_sampling("esc10_fold_1.hdf5")
    mismatch_first_farthest_traversal("esc10_fold_1.hdf5")
    
