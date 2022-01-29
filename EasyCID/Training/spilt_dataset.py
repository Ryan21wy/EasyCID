import numpy as np


def spilt(X, Y1):
    for i in range(X.shape[0]):
        X[i, :] = (X[i, :] - np.min(X[i, :])) / (np.max(X[i, :]) - np.min(X[i, :]))
    Xtrain = X[0:int(0.9 * X.shape[0])]
    Xtest = X[int(0.9 * X.shape[0]):X.shape[0]]
    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], 1)
    Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], 1)

    Ytrain = Y1[0:int(0.9 * X.shape[0])]
    Ytest = Y1[int(0.9 * X.shape[0]):X.shape[0]]
    return Xtrain, Xtest, Ytrain, Ytest


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def spilt_dataset(spectra, labels):
    shuffled_dataset, shuffled_labels = randomize(spectra, labels)
    Xtrain, Xtest, Ytrain, Ytest = spilt(shuffled_dataset, shuffled_labels)
    return Xtrain, Xtest, Ytrain, Ytest