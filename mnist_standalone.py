from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
from ncp.datasets.mnist import load_mnist

def network(data, layer_sizes = [256, 256, 10]):
    '''
    Defines network topology 
    '''
    # Define neural network topology (in this case, a simple MLP)
    hidden = data[0]
    labels = data[1]
    for size in layer_sizes:
        hidden = tf.layers.dense(
                inputs = hidden,
                units = size,
                activation = tf.nn.leaky_relu
                )
    logits = hidden

    #computes the traditional cross-entropy loss, which we want to minimize over the in-distribution training data
    standard_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
            )

    #computes the ncp_loss, in this case simply the entropy, which we want to minimize over the out-of-distribution training data
    class_probabilities = tf.nn.softmax(logits)
    ncp_loss = tf.reduce_sum(-class_probabilities * tf.log(class_probabilities))
    return standard_loss, ncp_loss, logits, class_probabilities

def generate_partial_mnist(digits_to_omit):
    '''
    Downloads and loads the mnist dataset into numpy arrays, and 
    removes the digits specified in the list (digits_to_omit).
    '''
    images, labels = load_mnist()
    for digit in digits_to_omit:
        indices = np.argwhere(labels == digit)
        labels = np.delete(labels, indices, axis = 0)
        images = np.delete(images, indices, axis = 0)
    return images, labels

def generate_od_data(images, labels):
    '''
    Applies some transformation to the data, to make it out-of-distribution
    '''
    od_images = images.copy()
    od_labels = labels.copy()
    # TODO
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

###################
# Some hyperparameters
learning_rate = 0.001
training_epochs = 1
batch_size = 100
display_step = 1
layer_sizes = [256, 256, 10]
digits_to_omit = []
alpha = 1 # weight factor between both contributions to the loss
##################

# PLACEHOLDERS FOR TRAINING DATA (id == in-distribution, od == out-of-distribution)
id_images_ = tf.placeholder(tf.float32, [None, 28 * 28])
id_labels_ = tf.placeholder(tf.int32, [None,])
id_data = (
        id_images_,
        tf.one_hot(id_labels_, 10)
           )

od_images_ = tf.placeholder(tf.float32, [None, 28 * 28])
od_labels_ = tf.placeholder(tf.int32, [None,])
od_data = (
        od_images_, tf.one_hot(od_labels_,
        10 - len(digits_to_omit))
          )

# need to specify template in order to ensure network variables are shared between id and od calculations
network_tpl = tf.make_template('network', network, layer_sizes = layer_sizes)
id_loss, _, logits, _ = network_tpl(id_data) # calculate CE loss for id input data
_, od_loss, logits_2, class_probabilities = network_tpl(od_data) # calculate entropy for od input data

# loss function is sum of id and od contributions
loss = alpha * id_loss + (1 - alpha) * od_loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Get in-distribution images and labels, with some digits ommited as specified in 'digits_to_omit' list.
    id_images, id_labels = generate_partial_mnist(digits_to_omit)
    # Generate out-of-distribution images
    od_images, od_labels = generate_od_data(id_images, id_labels) 
    id_batches = get_batches(id_images, id_labels, batch_size)
    od_batches = get_batches(od_images, od_labels, batch_size)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(id_labels) / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            id_batch_images, id_batch_labels = next(id_batches)
            od_batch_images, od_batch_labels = next(od_batches)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, class_probs, logits_ = sess.run(
                    [train_op, loss, class_probabilities, logits_2],
                    feed_dict={id_images_: id_batch_images,
                               id_labels_: id_batch_labels,
                               od_images_: od_batch_images,
                               od_labels_: od_batch_labels})
            # Compute average loss
            avg_cost += c / total_batch
        print(logits_[0])
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(tf.one_hot(id_labels_, 10), 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({id_images_: id_images, id_labels_: id_labels}))
