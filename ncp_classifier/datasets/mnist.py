import numpy as np
from urllib import request
import gzip
import pickle
import os

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
    ]


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        request.urlretrieve(
                base_url + name[1],
                os.path.join('ncp_classifier', 'datasets', name[1])
                )
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(
                os.path.join(
                    'ncp_classifier',
                    'datasets',
                    name[1]
                    ), 'rb') as f:
            mnist[name[0]] = np.frombuffer(
                    f.read(),
                    np.uint8,
                    offset=16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open(
                os.path.join(
                    'ncp_classifier',
                    'datasets',
                    name[1]), 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("./ncp_classifier/datasets/mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def init():
    download_mnist()
    save_mnist()


def load_mnist():
    # download_mnist()
    # save_mnist()
    with open("./ncp_classifier/datasets/mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    training_images = mnist['training_images']
    training_labels = mnist['training_labels']
    return training_images, training_labels


if __name__ == '__main__':
    init()
