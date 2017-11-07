import numpy as np
import os
from scipy.io import loadmat
import scipy
import sys
import cPickle as pkl
import tensorbayes as tb
from itertools import izip

def u2t(x):
    return x.astype('float32') / 255 * 2 - 1

def s2t(x):
    return x * 2 - 1

class Data(object):
    def __init__(self, images, labels=None, labeler=None, cast=False):
        self.images = images
        self.labels = labels
        self.labeler = labeler
        self.cast = cast

    def next_batch(self, bs):
        idx = np.random.choice(len(self.images), bs, replace=False)
        x = u2t(self.images[idx]) if self.cast else self.images[idx]
        y = self.labeler(x) if self.labels is None else self.labels[idx]
        return x, y

class MnistBias(object):
    def __init__(self, shape=(32, 32, 3)):
        mnist = Mnist(shape)
        trainx, trainy, trainz = self.create_data(mnist.train)
        self.train = Data(trainx, trainy)
        self.train_orig = Data(trainx, trainz)

        testx, testy, testz = self.create_data(mnist.test)
        self.test = Data(testx, testy)
        self.test_orig = Data(testx, testz)

    def create_data(self, data):
        # Select set of 0's, 1's, 2's
        x0, z0 = self.select_label(data, 0)
        x1, z1 = self.select_label(data, 1)
        x2, z2 = self.select_label(data, 2)

        x = np.concatenate((x0, x1, x0, x2), axis=0)
        z = np.concatenate((z0, z1, z0, z2), axis=0)
        n0, n1 = len(x0) + len(x1), len(x0) + len(x2)
        y = np.array([0] * n0 + [1] * n1)
        y = np.eye(2)[y]

        return x, y, z

    def select_label(self, data, label):
        idx = data.labels.argmax(axis=1) == label
        x = data.images[idx]
        y = data.labels[idx]
        return x, y

class Mnist(object):
    def __init__(self, shape=(32, 32, 3)):
        print "Loading MNIST"
        sys.stdout.flush()
        path = './data'
        data = np.load(os.path.join(path, 'mnist.npz'))
        trainx = np.concatenate((data['x_train'], data['x_valid']), axis=0)
        trainy = np.concatenate((data['y_train'], data['y_valid']))
        trainy = np.eye(10)[trainy].astype('float32')

        testx = data['x_test']
        testy = data['y_test'].astype('int')
        testy = np.eye(10)[testy].astype('float32')

        print "Resizing and norming"
        print "Original MNIST:", (trainx.min(), trainx.max()), trainx.shape
        sys.stdout.flush()
        trainx = self.resize_norm(trainx, shape)
        testx = self.resize_norm(testx, shape)
        print "New MNIST:", (trainx.min(), trainx.max()), trainx.shape

        self.train = Data(trainx, trainy)
        self.test = Data(testx, testy)

    @staticmethod
    def resize_norm(x, shape):
        H, W, C = shape
        x = x.reshape(-1, 28, 28)

        if x.shape[1:3] == (H, W):
            resized_x = s2t(x)

        else:
            resized_x = np.empty((len(x), H, W), dtype='float32')

            for i, img in enumerate(x):
                resized_x[i] = u2t(scipy.misc.imresize(img, (H, W)))

        resized_x = resized_x.reshape(-1, H, W, 1)
        resized_x = np.tile(resized_x, (1, 1, 1, C))
        return resized_x
