import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from ncp_classifier.datasets.mnist import load_mnist


def generate_partial_mnist(digits_to_omit):
    '''
    Downloads and loads the mnist dataset into numpy arrays, and 
    removes the digits specified in the list (digits_to_omit).
    '''
    images, labels = load_mnist()
    # Remove unwanted digits from training data
    indices = np.argwhere(labels == digits_to_omit[0])
    for digit in digits_to_omit[1:]:
        indices = np.concatenate((indices, np.argwhere(labels == digit)), axis=0)

    # Store the images and labels of the omitted training data, to be able to monitor the entropy
    # during NCP training. The network architecture does not allow to calculate the loss
    # (not enough units in output layer)
    om_images = images[indices.flatten()].copy()
    om_labels = labels[indices.flatten()].copy()
    labels = np.delete(labels, indices, axis=0)
    images = np.delete(images, indices, axis=0)

    # remap the other digits to {0, .., n_digits}, to be able to do one_hot encoding
    subtract_vector = np.zeros(10)
    for digit in digits_to_omit:
        subtract_vector[digit + 1:] += np.ones(10 - digit - 1)
    labels -= subtract_vector[labels].astype(np.uint8)
    return images, labels, om_images, om_labels


def generate_od_data(images, labels, transformations, plot=False):
    '''
    Applies some transformation to the data, to make it out-of-distribution
    transformations - dict of transformations to apply.
    Keys are the transformatinos
    values are the parameters.
    {}
    '''
    N = images.shape[0]

    # Leave labels the same
    od_labels = labels.copy()
    # Transform the images to generate OOD data
    if not transformations:
        od_images = images.copy()
    else:
        _ = [i.reshape(28, 28) for i in images]
        for T, amt in transformations.items():
            if T == 'rotate':
                # uniformly at random within [min,max]:
                if len(amt) == 2:
                    angles = np.random.uniform(amt[0], amt[1], size=N)
                # fixed value:
                elif len(amt) == 1:
                    angles = np.repeat(amt[0], N)
                od_images = np.concatenate(
                        [np.expand_dims(transform.rotate(_[i], angles[i], resize=False), axis=0) for i in range(N)],
                        axis=0)
            elif T == 'translate':
                pass
        # Visually check a few N random samples:
        if plot:
            M = 10
            print(od_images.shape)
            for i in range(M):
                plt.figure()
                plt.imshow(od_images[i])
                plt.title(str(od_labels[i]))
                plt.show()
        # Put back in the flattened shape expected in later code:
        od_images = od_images.reshape(od_images.shape[0], -1)
    return od_images, od_labels


def get_batches(images, labels, batch_size):
    '''
    Iterator which generates the batches for training
    '''
    n_batches = int(len(labels) / batch_size)
    while True:
        for i in range(n_batches):
            images_ = images[i * batch_size : (i + 1) * batch_size]
            labels_ = labels[i * batch_size : (i + 1) * batch_size]
            yield images_, labels_
