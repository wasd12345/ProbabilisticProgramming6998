from __future__ import print_function
import tensorflow as tf
import pickle
import os
import numpy as np
from random import seed
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp

from ncp_classifier.models.mnist_utils import generate_partial_mnist, generate_od_data, get_batches


##################
# Some hyperparameters
learning_rate = 7e-4
training_epochs = 20
batch_size = 100
display_step = 1
NCP_SCALE = 1 #scaling of logits output before the entropy is calculated, to reduce the size of the gradients
clip_at = (-10, 10) #gradient clipping
RANDOM_SEED = 11282018
NORMALIZE_ENTROPY = False #True #False
##################


def network(data, layer_sizes = [256, 256], ncp_scale = 0.1):
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
    weight_std = 0.1
    init_std = np.log(np.exp(weight_std) - 1).astype(np.float32)
    kernel_posterior = tfd.Independent(tfd.Normal(
        tf.get_variable(
            'kernel_mean', (hidden.shape[-1].value, layer_sizes[-1]), tf.float32,
            tf.random_normal_initializer(0, weight_std)), #initializes mean of weights in final layer. This is the only 'bayesian' layer.
        tf.nn.softplus(tf.get_variable(
            'kernel_std', (hidden.shape[-1].value, layer_sizes[-1]), tf.float32,
            tf.constant_initializer(init_std)))), 2) #initializes std of weights in final layer
    kernel_prior = tfd.Independent(tfd.Normal(
        tf.zeros_like(kernel_posterior.mean()),
        tf.zeros_like(kernel_posterior.mean()) + tf.nn.softplus(init_std)), 2)

    bias_prior = None
    bias_posterior = tfd.Deterministic(tf.get_variable(
        'bias_mean', (layer_sizes[-1],), tf.float32, tf.constant_initializer(0.0)))
    #tf.add_to_collection(
    #    tf.GraphKeys.REGULARIZATION_LOSSES,
    #    tfd.kl_divergence(kernel_posterior, kernel_prior)) #adds loss to collection
    logits = tfp.layers.DenseReparameterization( #ensures the kernels and biases are drawn from distributions
        layer_sizes[-1],
        kernel_prior_fn=lambda *args, **kwargs: kernel_prior,
        kernel_posterior_fn=lambda *args, **kwargs: kernel_posterior,
        bias_prior_fn=lambda *args, **kwargs: bias_prior,
        bias_posterior_fn=lambda *args, **kwargs: bias_posterior)(hidden)

    #logits = tf.layers.dense(inputs = hidden, units = layer_sizes[-1], activation = None)
    ##computes the traditional cross-entropy loss, which we want to minimize over the in-distribution training data
    standard_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
            )
    #logits = logits - tf.reduce_mean(logits)
    #class_probabilities = tf.nn.softmax(logits * tf.constant(ncp_scale, dtype = tf.float32))
    #mean, variance = tf.nn.moments(-class_probabilities * tf.log(tf.clip_by_value(class_probabilities, 1e-20, 1)), axes = [1])
    #ncp_loss = tf.reduce_mean(mean)

    logits = logits - tf.reduce_mean(logits)
    class_probabilities = tf.nn.softmax(logits * tf.constant(ncp_scale, dtype = tf.float32))
    entropy = -class_probabilities * tf.log(tf.clip_by_value(class_probabilities, 1e-20, 1))
    #Use the normalized entropy (divide by log_b(K) ) so is on [0,1] 
    #so easier to compare across experiments:
    if NORMALIZE_ENTROPY==True:
        baseK = tf.constant(layer_sizes[-1], dtype=tf.float32, shape=(layer_sizes[-1],))
        entropy /= tf.log(baseK)
    mean, variance = tf.nn.moments(entropy, axes = [1])
    ncp_loss = tf.reduce_mean(mean)
    ncp_std = tf.reduce_mean(tf.math.sqrt(variance))
    return standard_loss, ncp_loss, logits, ncp_std


#def network(data, layer_sizes, ncp_scale = 0.1):
#    '''
#    Defines network topology 
#    '''
#    # Define neural network topology (in this case, a simple MLP)
#    hidden = data[0]
#    labels = data[1]
#    for size in layer_sizes[:-1]:
#        hidden = tf.layers.dense(
#                inputs = hidden,
#                units = size,
#                activation = tf.nn.leaky_relu
#                )
#    logits = tf.layers.dense(inputs = hidden, units = layer_sizes[-1], activation = None)
#    #computes the traditional cross-entropy loss, which we want to minimize over the in-distribution training data
#    standard_loss = tf.reduce_mean(
#            tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
#            )
#    #computes the ncp_loss, in this case simply the entropy, which we want to minimize over the out-of-distribution training data
#    logits = logits - tf.reduce_mean(logits)
#    class_probabilities = tf.nn.softmax(logits * tf.constant(ncp_scale, dtype = tf.float32))
#    entropy = -class_probabilities * tf.log(tf.clip_by_value(class_probabilities, 1e-20, 1))
#    #Use the normalized entropy (divide by log_b(K) ) so is on [0,1] 
#    #so easier to compare across experiments:
#    if NORMALIZE_ENTROPY==True:
#        baseK = tf.constant(layer_sizes[-1], dtype=tf.float32, shape=(layer_sizes[-1],))
#        entropy /= tf.log(baseK)
#    mean, variance = tf.nn.moments(entropy, axes = [1])
#    ncp_loss = tf.reduce_mean(mean)
#    ncp_std = tf.reduce_mean(tf.math.sqrt(variance))
#    return standard_loss, ncp_loss, logits, class_probabilities, ncp_std





def run_single(digits_to_omit, ood_transformations, alpha, experiment_suffix):
    """
    Run a single experiment, i.e. train one network to completion on one dataset
    """
    
    logging = dict()
    logging['log_step'] = 100000
    logging['training_epochs'] = training_epochs
    logging['id_ncp_loss'] = [] #entropy (uncertainty) loss
    logging['id_ncp_std'] = []
    logging['id_loss'] = [] #standard CE loss
    logging['od_loss'] = [] #standard CE loss
    logging['od_ncp_loss'] = [] #entropy (uncertainty) loss
    logging['om_ncp_loss'] = [] #entropy (uncertainty) loss
    logging['om_ncp_std'] = []   
    logging['od_ncp_std'] = []
    
    logging['ncp_scale'] = NCP_SCALE
    logging['alpha'] = alpha
    logging['clip_at'] = clip_at
  
    logging['learning_rate'] = learning_rate
    logging['batch_size'] = batch_size
    logging['display_step'] = display_step
    logging['RANDOM_SEED'] = RANDOM_SEED
    
    logging['digits_to_omit'] = digits_to_omit.copy()     
    logging['output_layer_size'] = 10 - len(digits_to_omit)
    logging['layer_sizes'] = [256, 256, 200, 50, logging['output_layer_size']]
    
    
    
    # PLACEHOLDERS FOR TRAINING DATA (id == in-distribution, od == out-of-distribution)
    id_images_ = tf.placeholder(tf.float32, [None, 28 * 28])
    id_labels_ = tf.placeholder(tf.int32, [None,])
#    one_hot_ = tf.one_hot(id_labels_, output_layer_size)
    id_data = (
            id_images_,
            tf.one_hot(id_labels_, logging['output_layer_size'])
               )
    
    od_images_ = tf.placeholder(tf.float32, [None, 28 * 28])
    od_labels_ = tf.placeholder(tf.int32, [None,])
    od_data = (
            od_images_,
            tf.one_hot(od_labels_, logging['output_layer_size'])
              )
    
    # need to specify template in order to ensure network variables are shared between id and od calculations
    network_tpl = tf.make_template('network', network, layer_sizes=logging['layer_sizes'], ncp_scale=logging['ncp_scale'])
    id_loss, id_ncp_loss, id_logits, id_ncp_std  = network_tpl(id_data) # calculate CE loss for id input data
    od_loss, od_ncp_loss, od_logits, od_ncp_std = network_tpl(od_data) # calculate entropy for od input data
    
    # loss function is sum of id standard loss and od ncp loss
    loss = alpha * id_loss - (1 - alpha) * od_ncp_loss
    
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
        od_images, od_labels = generate_od_data(id_images, id_labels, ood_transformations, plot=False) 
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
                
                
                _, id_loss_, id_ncp_loss_, od_loss_, od_ncp_loss_, id_ncp_std_, od_ncp_std_ = sess.run(
                        [train_op, id_loss, id_ncp_loss, od_loss, od_ncp_loss, id_ncp_std, od_ncp_std],
                        feed_dict = {id_images_: id_batch_images,
                                    id_labels_: id_batch_labels,
                                    od_images_: od_batch_images,
                                    od_labels_: od_batch_labels})
                #om_entropy_, om_entropy_std_ = sess.run([id_ncp_loss,id_ncp_std],
                #        feed_dict = {id_images_: om_images,
                #                    id_labels_: om_labels,
                #                    od_images_: od_batch_images,
                #                    od_labels_: od_batch_labels}) #Only interested in entropy of omitted data            
            
                
                # Compute average standard loss
                avg_cost += id_loss_ / total_batch
                counter += 1
    
                if counter % logging['log_step'] == 0:
                    #Standard cross-entropy losses
                    logging['id_loss'].append(id_loss_)
                    logging['od_loss'].append(od_loss_)
                    #Uncertainty (entropy) losses
                    logging['id_ncp_loss'].append(id_ncp_loss_)
                    logging['od_ncp_loss'].append(od_ncp_loss_)
                    #logging['om_ncp_loss'].append(om_entropy_)
                    #Std's pf those uncertainty (entropy) losses
                    logging['id_ncp_std'].append(id_ncp_std_)
                    logging['od_ncp_std'].append(od_ncp_std_)
                    #logging['om_ncp_std'].append(om_entropy_std_)
                    
                    print('id entropy loss: ',id_ncp_loss_, '   id entropy std: ', id_ncp_std_)
                    print('od entropy loss: ',od_ncp_loss_, '   od entropy std: ', od_ncp_std_)
                    #print('om entropy loss: ',om_entropy_, '   om entropy std: ', om_entropy_std_)
                    print()
    
            #print(logits_[0])
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
        print("Optimization Finished!")
        om_entropy_, om_entropy_std_ = sess.run([id_ncp_loss,id_ncp_std],
                feed_dict = {id_images_: om_images,
                            id_labels_: om_labels,
                            od_images_: od_batch_images,
                            od_labels_: od_batch_labels}) #Only interested in entropy of omitted data            
        logging['om_ncp_loss'].append(om_entropy_)
        logging['om_ncp_std'].append(om_entropy_std_)
    

        # Test model
        pred = tf.nn.softmax(id_logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(tf.one_hot(id_labels_, logging['output_layer_size']), 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        logging['id_acc'] = accuracy.eval({id_images_: id_images, id_labels_: id_labels})
        #Same for OOD data:
        od_pred = tf.nn.softmax(od_logits)
        od_correct_prediction = tf.equal(tf.argmax(od_pred, 1), tf.argmax(tf.one_hot(od_labels_, logging['output_layer_size']), 1))
        od_accuracy = tf.reduce_mean(tf.cast(od_correct_prediction, "float"))
        logging['od_acc'] = od_accuracy.eval({od_images_: od_images, od_labels_: od_labels})
        
        print("Full id_acc:", logging['id_acc'])
        print("Full od_acc:", logging['od_acc'])
        
        #This should always be 0 for an unseen digit since our classifier can never choose this label
#        print("Full om_acc:", logging['om_acc'])
            
        
        #Calculate mean + std of entropy over all 3 full datasets:
        _, id_entropy_mean, od_entropy_mean, id_entropy_std, od_entropy_std = sess.run(
                [train_op, id_ncp_loss, od_ncp_loss, id_ncp_std, od_ncp_std],
                feed_dict = {id_images_: id_images,
                            id_labels_: id_labels,
                            od_images_: od_images,
                            od_labels_: od_labels})
        
        logging['id_entropy_mean'] = id_entropy_mean
        logging['id_entropy_std'] = id_entropy_std
        logging['od_entropy_mean'] = od_entropy_mean
        logging['od_entropy_std'] = od_entropy_std        
        #"om_ncp_loss" and "om_ncp_std" are already the whole set values for om
        
        
        logdir = os.path.join('ncp_classifier', 'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        logpath = os.path.join(logdir, f"log_ncp_on__{experiment_suffix}.p")
        pickle.dump(logging, open(logpath, "wb" ) )




def rotation_experiment():
    """
    Sweep through a range of transformations.
    (In this case, rotation angles)
    """
    #Experiment parameters:
    EXPERIMENT_NAME = 'rotate'
    ROTATIONS_UPPER_BOUND = [i*10. for i in range(10)]
    DIGITS_TO_OMIT = [8,9]
    ALPHA = 1e-4 # weight factor between both contributions to the loss
    
    for i, UB in enumerate(ROTATIONS_UPPER_BOUND):
        #For each experiment: seed -> same weights init, same data splits
        seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        tf.random.set_random_seed(RANDOM_SEED)
        
        experiment_suffix = f"{EXPERIMENT_NAME}_{i}" 
        ood_transformations = {'rotate':[0.,UB],
                               #'translate':None,
                               #'scale':[],
                               #'affine_random':None,
                               #'perspective_random':None,
                               #'swirl':[],
                               #'noise':None,
                               }        
        run_single(DIGITS_TO_OMIT, ood_transformations,ALPHA,experiment_suffix)
        tf.reset_default_graph()



def alpha_experiment():
    """
    Sweep through a range of alpha values in loss function
    (tradeoff btwn standard cross-entropy loss vs. OOD uncertainty loss)
    """
    #Experiment parameters:
    EXPERIMENT_NAME = 'alpha'
    alpha_list = np.logspace(-7., 0., 8)
    DIGITS_TO_OMIT = [8,9]
    
    for i, a in enumerate(alpha_list):
        print('alpha:', a)
        #For each experiment: seed -> same weights init, same data splits
        seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        tf.random.set_random_seed(RANDOM_SEED)
        
        experiment_suffix = f"{EXPERIMENT_NAME}_{i}"
        ALPHA = alpha_list[i]
        THETA = 60.
        ood_transformations = {'rotate':[-THETA,THETA]}
        
        run_single(DIGITS_TO_OMIT,ood_transformations,ALPHA,experiment_suffix)
        tf.reset_default_graph()
        


def digits_out_experiment():
    """
    Successively hold out more digit classes, e.g.
    set 1: in-distribution=[0,...,8] and OOD=[9]
    set 2: in-distribution=[0,...,7] and OOD=[8,9]
    set 3: in-distribution=[0,...,3] and OOD=[4,5,6,7,8,9]
    """
    #Must have at least 1 hold out digit class, 
    #and at least 2 in-distribution digit classes
    DIG_OUTS = [[5, 6, 2],
                #[8,9],
                #[4,5,6,7,8,9],
                #[2,3,4,5,6,7,8,9]
                ]
    
    #Experiment parameters:
    EXPERIMENT_NAME = 'digout'
    alpha_list = np.logspace(-7., -4., 5)
    for i, d_ood in enumerate(DIG_OUTS):
        digs = ''
        for mm in d_ood:
            digs += str(mm)
        digs += 'out'
        
        for i, a in enumerate(alpha_list):
            print('alpha:', a)
            #For each experiment: seed -> same weights init, same data splits
            seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            tf.random.set_random_seed(RANDOM_SEED)
            
            experiment_suffix = f"{EXPERIMENT_NAME}_{digs}_{i}"
            ALPHA = alpha_list[i]
            THETA = 90.
            ood_transformations = {'rotate':[45,135]}
            
            run_single(d_ood, ood_transformations, ALPHA, experiment_suffix)
            tf.reset_default_graph()



if __name__=="__main__":
    
    #Do the experiment with various angles of rotations:
#    rotation_experiment()
    
    #Do the alpha loss function experiment
#    alpha_experiment()
    
    #Do the successive hold out digits experiment
    digits_out_experiment()
    
    #Do experiment with ...
