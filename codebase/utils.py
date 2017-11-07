from args import args
import tensorflow as tf
import shutil
import os
import datasets
import numpy as np

def delete_existing(path):
    if args.run < 999:
        assert not os.path.exists(path), "Cannot overwrite {:s}".format(path)

    else:
        if os.path.exists(path):
            shutil.rmtree(path)

def get_data(name):
    if name == 'mnist32':
        return datasets.Mnist(shape=(32, 32, 1))

    if name == 'mnistbias':
        return datasets.MnistBias(shape=(32, 32, 1))

    else:
        raise Exception('dataset {:s} not recognized'.format(name))
