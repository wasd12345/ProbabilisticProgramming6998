# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow_probability import distributions as tfd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ncp import tools


def network(inputs, config):
  '''
  Based on bbb.py and the example in the documentation for tfp.layers.DenseReparameterization
  This implementation is super super super sloppy
  '''
  init_std = np.log(np.exp(config.weight_std) - 1).astype(np.float32)
  hidden = inputs
  # Define hidden normal layers according to config.layer_sizes
  model = tf.keras.Sequential()
  for size in config.layer_sizes:
    model.add(tf.keras.layers.Dense(size, activation = tf.nn.leaky_relu))

  # Define hidden bayesian layers according to config.bayesian_layer_sizes.
  # Still need to add specification of the normal prior parameters of the weights/biases
  # But how? create n = len(config.bayesian_layer_sizes) distributions kernel_prior_i?
  for size in config.bayesian_layer_sizes:
    model.add(tfp.layers.DenseReparameterization(size, activation = tf.nn.leaky_relu))

  # Add final softmax layer
  model.add(tfp.layers.DenseReparameterization(10, activation = 'softmax'))

  # Add KL loss to collection
  tf.add_to_collection(
      tf.GraphKeys.REGULARIZATION_LOSSES,
      model.losses) #adds loss to collection

  logits = model(inputs)
  return logits, model


def define_graph(config):
  network_tpl = tf.make_template('network', network, config=config)
  inputs = tf.placeholder(tf.float32, [None, config.num_inputs])
  targets = tf.placeholder(tf.float32, [None, 1])
  num_visible = tf.placeholder(tf.int32, [])
  batch_size = tf.shape(inputs)[0]
  logits, model = network_tpl(inputs)
  neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
      labels = targets, logits = logits)

  #data_dist, mean_dist = network_tpl(inputs) #output from network
  assert len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  divergence = sum([
      tf.reduce_sum(tensor) for tensor in
      tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)])
  num_batches = tf.to_float(num_visible) / tf.to_float(batch_size)
  losses = [config.divergence_scale * divergence / num_batches,
      neg_log_likelihood]
  
  loss = sum(tf.reduce_sum(loss) for loss in losses) / tf.to_float(batch_size)
  optimizer = tf.train.AdamOptimizer(config.learning_rate)
  gradients, variables = zip(*optimizer.compute_gradients(
      loss, colocate_gradients_with_ops=True))
  if config.clip_gradient:
    gradients, _ = tf.clip_by_global_norm(gradients, config.clip_gradient)
  optimize = optimizer.apply_gradients(zip(gradients, variables))
  #data_mean = mean_dist.mean()
  #data_noise = data_dist.stddev()
  #data_uncertainty = mean_dist.stddev()
  return tools.AttrDict(locals())
