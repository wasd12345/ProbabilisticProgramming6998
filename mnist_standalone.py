from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
from ncp.datasets.mnist import load_mnist
import pickle

def network(data, layer_sizes = [256, 256, 10], ncp_scale = 0.1):
    '''
    Defines network topology 
    '''
    # Define neural network topology (in this case, a simple MLP)
    hidden = data[0]
    labels = data[1]
    for size in layer_sizes[:-1]:
        hidden = tf.layers.dense(
                inputs = hidden,
                units = size,
                activation = tf.nn.leaky_relu
                )
    logits = tf.layers.dense(inputs = hidden, units = layer_sizes[-1], activation = None)

    #computes the traditional cross-entropy loss, which we want to minimize over the in-distribution training data
    standard_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
            )

    #computes the ncp_loss, in this case simply the entropy, which we want to minimize over the out-of-distribution training data
    logits = logits - tf.reduce_mean(logits)
    class_probabilities = tf.nn.softmax(logits * tf.constant(ncp_scale, dtype = tf.float32))
    ncp_loss = tf.reduce_mean(-class_probabilities * tf.log(tf.clip_by_value(class_probabilities, 1e-10, 1)))
    return standard_loss, ncp_loss, logits, class_probabilities

def generate_partial_mnist(digits_to_omit):
    '''
    Downloads and loads the mnist dataset into numpy arrays, and 
    removes the digits specified in the list (digits_to_omit).
    '''
    images, labels = load_mnist()
    # Remove unwanted digits from training data
    indices = np.argwhere(labels == digits_to_omit[0])
    for digit in digits_to_omit[1:]:
        indices = np.concatenate((indices, np.argwhere(labels == digit)), axis = 0)

    # Store the images and labels of the omitted training data, to be able to monitor the entropy
    # during NCP training. The network architecture does not allow to calculate the loss
    # (not enough units in output layer)
    om_images = images[indices.flatten()].copy()
    om_labels = labels[indices.flatten()].copy() 
    labels = np.delete(labels, indices, axis = 0)
    images = np.delete(images, indices, axis = 0)

    #remap the other digits to {0, .., n_digits}, to be able to do one_hot encoding
    subtract_vector = np.zeros(10)
    for digit in digits_to_omit:
        subtract_vector[digit + 1:] += np.ones(10 - digit - 1)
    labels -= subtract_vector[labels].astype(np.uint8) 
    return images, labels, om_images, om_labels

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

##################
# Some hyperparameters
learning_rate = 0.0001
training_epochs = 5
batch_size = 100
display_step = 1
digits_to_omit = [3, 5, ]
output_layer_size = 10 - len(digits_to_omit)
layer_sizes = [256, 256, output_layer_size]
ncp_scale = 1 #scaling of logits output before the entropy is calculated, to reduce the size of the gradients
alpha = 1 # weight factor between both contributions to the loss
clip_at = (-10, 10)

logging = dict()
logging['log_step'] = 100
logging['training_epochs'] = training_epochs
logging['id_ncp_loss'] = []
logging['id_loss'] = []
logging['od_loss'] = []
logging['od_ncp_loss'] = []
logging['om_entropy'] = []
logging['ncp_scale'] = ncp_scale
logging['alpha'] = alpha
logging['clip_at'] = clip_at
logging['digits_to_omit'] = digits_to_omit.copy()
##################

# PLACEHOLDERS FOR TRAINING DATA (id == in-distribution, od == out-of-distribution)
id_images_ = tf.placeholder(tf.float32, [None, 28 * 28])
id_labels_ = tf.placeholder(tf.int32, [None,])
one_hot_ = tf.one_hot(id_labels_, output_layer_size)
id_data = (
        id_images_,
        tf.one_hot(id_labels_, output_layer_size)
           )

od_images_ = tf.placeholder(tf.float32, [None, 28 * 28])
od_labels_ = tf.placeholder(tf.int32, [None,])
od_data = (
        od_images_,
        tf.one_hot(od_labels_, output_layer_size)
          )

# need to specify template in order to ensure network variables are shared between id and od calculations
network_tpl = tf.make_template('network', network, layer_sizes = layer_sizes, ncp_scale = ncp_scale)
id_loss, id_ncp_loss, logits, class_probabilities = network_tpl(id_data) # calculate CE loss for id input data
od_loss, od_ncp_loss, _, _ = network_tpl(od_data) # calculate entropy for od input data

# loss function is sum of id standard loss and od ncp loss
loss = alpha * id_loss #+ (1 - alpha) * od_ncp_loss

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad, clip_at[0], clip_at[1]), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)

# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Get in-distribution images and labels, with some digits ommited as specified in 'digits_to_omit' list.
    id_images, id_labels, om_images, om_labels = generate_partial_mnist(digits_to_omit)
    # Generate out-of-distribution images
    od_images, od_labels = generate_od_data(id_images, id_labels) 
    id_batches = get_batches(id_images, id_labels, batch_size)
    od_batches = get_batches(od_images, od_labels, batch_size)

    counter = 0
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(id_labels) / batch_size)
        # Loop over all batches
        for i in range(total_batch):
        #for i in range(3):
            id_batch_images, id_batch_labels = next(id_batches)
            od_batch_images, od_batch_labels = next(od_batches)

            _, id_loss_, id_ncp_loss_, od_loss_, od_ncp_loss_, class_ = sess.run(
                    [train_op, id_loss, id_ncp_loss, od_loss, od_ncp_loss, class_probabilities],
                    feed_dict = {id_images_: id_batch_images,
                                id_labels_: id_batch_labels,
                                od_images_: od_batch_images,
                                od_labels_: od_batch_labels})
            #om_entropy_ = sess.run([id_ncp_loss],
            #        feed_dict = {id_images_: om_images,
            #                    id_labels_: om_labels,
            #                    od_images_: od_batch_images,
            #                    od_labels_: od_batch_labels}) #Only interested in entropy of omitted data
            om_entropy_ = 1

            # Compute average standard loss
            avg_cost += id_loss_ / total_batch
            counter += 1

            if counter % logging['log_step'] == 0:
                logging['id_loss'].append(id_loss_)
                logging['id_ncp_loss'].append(id_ncp_loss_)
                logging['od_loss'].append(od_loss_)
                logging['od_ncp_loss'].append(od_ncp_loss_)
                logging['om_entropy'].append(om_entropy_)
                print('ncp loss: ',id_ncp_loss_)
                #print('class probabilities: ', class_)

        #print(logits_[0])
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(tf.one_hot(id_labels_, output_layer_size), 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({id_images_: id_images, id_labels_: id_labels}))
    pickle.dump(logging, open( "log.p", "wb" ) )
