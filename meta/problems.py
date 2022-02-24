# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
# ==============================================================================
"""Learning 2 Learn problems."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tarfile
import sys

import nilmtk

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
import tensorflow as tf
import pdb
import tensorflow_datasets as tfds
from vgg16 import VGG16
import tensorflow_probability as tfp

from nilmtk import *

tfd = tfp.distributions

_nn_initializers = {
    "w": tf.random_normal_initializer(mean=0, stddev=0.01),
    "b": tf.random_normal_initializer(mean=0, stddev=0.01),
}

def _xent_loss(output, labels):
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output,
                                                        labels=labels)
  return tf.reduce_mean(loss)


def _maybe_download_cifar10(path):
  """Download and extract the tarball from Alex's website."""
  if not os.path.exists(path):
    os.makedirs(path)
  filepath = os.path.join(path, CIFAR10_FILE)
  if not os.path.exists(filepath):
    print("Downloading CIFAR10 dataset to {}".format(filepath))
    url = os.path.join(CIFAR10_URL, CIFAR10_FILE)
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    print("Successfully downloaded {} bytes".format(statinfo.st_size))
    tarfile.open(filepath, "r:gz").extractall(path)
    
#TODO ganze nilm util methoden auslagern
def _get_mains_and_subs(datasets, appliances, power, drop_nans, sample_period=1, artificial_aggregate=False):

    # This function has a few issues, which should be addressed soon
    print("............... Loading Data for training ...................")
    # store the train_main readings for all buildings
    train_mains = []
    train_submeters = [[] for i in range(len(appliances))]
    for dataset in datasets:
        print("Loading data for ",dataset, " dataset")
        train=DataSet(datasets[dataset]['path'])
        for building in datasets[dataset]['buildings']:
            print("Loading building ... ",building)
            train.set_window(start=datasets[dataset]['buildings'][building]['start_time'],
                             end=datasets[dataset]['buildings'][building]['end_time'])
            train_df = next(train.buildings[building].elec.mains().load(physical_quantity='power', 
                                                                        ac_type=power['mains'], 
                                                                        sample_period=sample_period))
            train_df = train_df[[list(train_df.columns)[0]]]
            appliance_readings = [] # List of appliance dataframes
            
            # TODO does meta expect a different format here with indeces?

            # get and append all single appliance dfs
            for appliance_name in appliances:
                appliance_df = next(train.buildings[building].elec[appliance_name].load(
                    physical_quantity='power', ac_type=power['appliance'], sample_period=sample_period))
                appliance_df = appliance_df[[list(appliance_df.columns)[0]]]
                appliance_readings.append(appliance_df)

            if drop_nans:
                train_df, appliance_readings = _dropna(train_df, appliance_readings)

            if artificial_aggregate:
                print ("Creating an Artificial Aggregate")
                train_df = pd.DataFrame(np.zeros(appliance_readings[0].shape),index =
                                        appliance_readings[0].index,columns=appliance_readings[0].columns)
                for app_reading in appliance_readings:
                    train_df+=app_reading

            train_mains.append(train_df)
            for i,appliance_name in enumerate(appliances):
                train_submeters[i].append(appliance_readings[i])

    appliance_readings = []
    for i,appliance_name in enumerate(appliances):
        appliance_readings.append((appliance_name, train_submeters[i]))

    train_submeters = appliance_readings   

    return train_mains,train_submeters

def _dropna(mains_df, appliance_dfs=[]):
        """
        Drops the missing values in the Mains reading and appliance readings and returns consistent data by copmuting the intersection
        """
        print ("Dropping missing values")

        # The below steps are for making sure that data is consistent by doing intersection across appliances
        mains_df = mains_df.dropna()
        ix = mains_df.index
        mains_df = mains_df.loc[ix]
        for i in range(len(appliance_dfs)):
            appliance_dfs[i] = appliance_dfs[i].dropna()
    
        for  app_df in appliance_dfs:
            ix = ix.intersection(app_df.index)
        mains_df = mains_df.loc[ix]
        new_appliances_list = []
        for app_df in appliance_dfs:
            new_appliances_list.append(app_df.loc[ix])
        return mains_df,new_appliances_list

@tf.function
def _call_preprocessing(mains_lst, submeters_lst, method, window_size):
    if method == 'train':
        mains_df_list = []
        for mains in mains_lst:
            new_mains = mains.values.flatten()
            n = window_size
            units_to_pad = n // 2
            new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
            new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
            new_mains = (new_mains - self.mains_mean) / self.mains_std
            mains_df_list.append(pd.DataFrame(new_mains))

        appliance_list = []
        for app_index, (app_name, app_df_list) in enumerate(submeters_lst):
            if app_name in self.appliance_params:
                app_mean = self.appliance_params[app_name]['mean']
                app_std = self.appliance_params[app_name]['std']
            else:
                print("Parameters for ", app_name, " were not found!")
                raise ApplianceNotFoundError()

            processed_appliance_dfs = []

            for app_df in app_df_list:
                new_app_readings = app_df.values.reshape((-1, 1))
                # This is for choosing windows
                new_app_readings = (new_app_readings - app_mean) / app_std
                # Return as a list of dataframe
                processed_appliance_dfs.append(pd.DataFrame(new_app_readings))
            appliance_list.append((app_name, processed_appliance_dfs))
        return mains_df_list, appliance_list

    else:
        mains_df_list = []

        for mains in mains_lst:
            new_mains = mains.values.flatten()
            n = self.sequence_length
            units_to_pad = n // 2
            new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
            new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
            new_mains = (new_mains - self.mains_mean) / self.mains_std
            mains_df_list.append(pd.DataFrame(new_mains))
        return mains_df_list
    
    
def cifar10(path,
            batch_norm=True,
            batch_size=128,
            num_threads=4,
            min_queue_examples=1000,
            mode="train"):
  """Cifar10 classification with a convolutional network."""

  # Data.
  _maybe_download_cifar10(path)
  if mode == "train":
    filenames = [os.path.join(path, CIFAR10_FOLDER, "data_batch_{}.bin".format(i)) for i in xrange(1, 6)]
  elif mode == "test":
    filenames = [os.path.join(path, CIFAR10_FOLDER, "test_batch.bin")]
  else:
    raise ValueError("Mode {} not recognised".format(mode))

  depth = 3
  height = 32
  width = 32
  label_bytes = 1
  image_bytes = depth * height * width
  record_bytes = label_bytes + image_bytes
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  _, record = reader.read(tf.train.string_input_producer(filenames))
  record_bytes = tf.decode_raw(record, tf.uint8)

  label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
  raw_image = tf.slice(record_bytes, [label_bytes], [image_bytes])
  image = tf.cast(tf.reshape(raw_image, [depth, height, width]), tf.float32)
  # height x width x depth.
  image = tf.transpose(image, [1, 2, 0])
  image = tf.math.divide(image, 255)

  queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                                min_after_dequeue=min_queue_examples,
                                dtypes=[tf.float32, tf.int32],
                                shapes=[image.get_shape(), label.get_shape()])
  enqueue_ops = [queue.enqueue([image, label]) for _ in xrange(num_threads)]
  tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

  def network(inputs, training=True):

      def _conv_activation(x):
          return tf.nn.max_pool(tf.nn.relu(x),
                                ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1],
                                padding="VALID")
    
      def conv_layer(inputs, strides, c_h, c_w, output_channels, padding, name):
          n_channels = int(inputs.get_shape()[-1])
          with tf.variable_scope(name) as scope:
              kernel1 = tf.get_variable('weights1',
                                        shape=[c_h, c_w, n_channels, output_channels],
                                        dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(stddev=0.01)
                                        )
            
              biases1 = tf.get_variable('biases1', [output_channels], initializer=tf.constant_initializer(0.0))
          inputs = tf.nn.conv2d(inputs, kernel1, [1, strides, strides, 1], padding)
          inputs = tf.nn.bias_add(inputs, biases1)
          if batch_norm:
              inputs = tf.layers.batch_normalization(inputs, training=training)
          inputs = _conv_activation(inputs)
          return inputs

      inputs = conv_layer(inputs, 2, 3, 3, 16, "VALID", 'conv_layer1')
      inputs = conv_layer(inputs, 2, 5, 5, 32, "VALID", 'conv_layer2')
      inputs = tf.reshape(inputs, [batch_size, -1])
      fc_shape2 = int(inputs.get_shape()[1])
      weights = tf.get_variable("fc_weights",
                                shape=[fc_shape2, 10],
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(stddev=0.01))
      bias = tf.get_variable("fc_bias",
                             shape=[10, ],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

      return tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, weights), bias))

  
  def build():
    image_batch, label_batch = queue.dequeue_many(batch_size)
    label_batch = tf.reshape(label_batch, [batch_size])
    output = network(image_batch)

    return _xent_loss(output, label_batch)

  return build


def NAS(path,
        batch_norm=True,
        batch_size=128,
        num_threads=4,
        min_queue_examples=1000,
        mode="train"):
    """Cifar10 classification with a convolutional network."""

    # Data.
    _maybe_download_cifar10(path)

    # Read images and labels from disk.
    if mode == "train":
        filenames = [os.path.join(path, CIFAR10_FOLDER, "data_batch_{}.bin".format(i)) for i in xrange(1, 6)]
    elif mode == "test":
        filenames = [os.path.join(path, CIFAR10_FOLDER, "test_batch.bin")]
    else:
        raise ValueError("Mode {} not recognised".format(mode))

    depth = 3
    height = 32
    width = 32
    label_bytes = 1
    image_bytes = depth * height * width
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, record = reader.read(tf.train.string_input_producer(filenames))
    record_bytes = tf.decode_raw(record, tf.uint8)

    label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    raw_image = tf.slice(record_bytes, [label_bytes], [image_bytes])
    image = tf.cast(tf.reshape(raw_image, [depth, height, width]), tf.float32)
    # height x width x depth.
    image = tf.transpose(image, [1, 2, 0])
    image = tf.div(image, 255)

    queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                                  min_after_dequeue=min_queue_examples,
                                  dtypes=[tf.float32, tf.int32],
                                  shapes=[image.get_shape(), label.get_shape()])
    enqueue_ops = [queue.enqueue([image, label]) for _ in xrange(num_threads)]
    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

    # Network
    def network(inputs, training=True):
        def conv_layer(inputs, strides, c_h, c_w, output_channels, padding, name):
            n_channels = int(inputs.get_shape()[-1])
            with tf.variable_scope(name) as scope:
                kernel1 = tf.get_variable('weights1',
                                          shape=[c_h, c_w, n_channels, output_channels],
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(stddev=0.01)
                                          )

                biases1 = tf.get_variable('biases1', [output_channels], initializer=tf.constant_initializer(0.0))
            inputs = tf.nn.conv2d(inputs, kernel1, [1, strides, strides, 1], padding)
            inputs = tf.nn.bias_add(inputs, biases1)
            if batch_norm:
                inputs = tf.layers.batch_normalization(inputs, training=training)
            inputs = tf.nn.relu(inputs)
            return inputs

        def _pooling(x):
            return tf.nn.avg_pool(x,
                                  ksize=[1, 3, 3, 1],
                                  strides=[1, 1, 1, 1],
                                  padding="SAME")

        node0 = conv_layer(inputs, 1, 3, 3, 16, "SAME", 'node0')
        node0_onto_node2 = conv_layer(node0, 1, 3, 3, 16, "SAME", 'node0_onto_node2')
        node1 = conv_layer(node0, 1, 3, 3, 16, "SAME", 'node1')
        node1_onto_node3 = conv_layer(node1, 1, 3, 3, 16, "SAME", 'node1_onto_node3')
        node2 = _pooling(node1) + node0_onto_node2
        node3 = node2 + node1_onto_node3 + node0
        node_final = tf.reduce_mean(tf.reshape(node3, [batch_size, -1, 16]), axis=1)

        fc_shape2 = int(node_final.get_shape()[1])
        weights = tf.get_variable("fc_weights",
                                  shape=[fc_shape2, 10],
                                  dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(stddev=0.01))
        bias = tf.get_variable("fc_bias",
                               shape=[10, ],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(node_final, weights), bias))

    def build():
        image_batch, label_batch = queue.dequeue_many(batch_size)
        label_batch = tf.reshape(label_batch, [batch_size])

        output = network(image_batch)
        return _xent_loss(output, label_batch)

    return build


def vgg16_cifar10(path,  # pylint: disable=invalid-name
            batch_norm=False,
            batch_size=128,
            num_threads=4,
            min_queue_examples=1000,
            mode="train"):
    """Cifar10 classification with a convolutional network."""
    
    # Data.
    _maybe_download_cifar10(path)
    # pdb.set_trace()
    # Read images and labels from disk.
    if mode == "train":
        filenames = [os.path.join(path,
                                  CIFAR10_FOLDER,
                                  "data_batch_{}.bin".format(i))
                     for i in xrange(1, 6)]
        is_training = True
    elif mode == "test":
        filenames = [os.path.join(path, CIFAR10_FOLDER, "test_batch.bin")]
        is_training = False
    else:
        raise ValueError("Mode {} not recognised".format(mode))
    
    depth = 3
    height = 32
    width = 32
    label_bytes = 1
    image_bytes = depth * height * width
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, record = reader.read(tf.train.string_input_producer(filenames))
    record_bytes = tf.decode_raw(record, tf.uint8)
    
    label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.int32)
    raw_image = tf.slice(record_bytes, [label_bytes], [image_bytes])
    image = tf.cast(tf.reshape(raw_image, [depth, height, width]), tf.float32)
    # height x width x depth.
    image = tf.transpose(image, [1, 2, 0])
    image = tf.math.divide(image, 255)

    queue = tf.RandomShuffleQueue(capacity=min_queue_examples + 3 * batch_size,
                                  min_after_dequeue=min_queue_examples,
                                  dtypes=[tf.float32, tf.int32],
                                  shapes=[image.get_shape(), label.get_shape()])
    enqueue_ops = [queue.enqueue([image, label]) for _ in xrange(num_threads)]
    tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

    vgg = VGG16(0.5, 10)
    def build():
        image_batch, label_batch = queue.dequeue_many(batch_size)
        label_batch = tf.reshape(label_batch, [batch_size])
        # pdb.set_trace()
        output = vgg._build_model(image_batch)
        # print(output.shape)
        return _xent_loss(output, label_batch)
    
    return build



def mnist(layers,
          activation="sigmoid",
          batch_size=128,
          mode="train"):
    """Mnist classification with a multi-layer perceptron."""
    initializers = _nn_initializers

    if activation == "sigmoid":
        activation_op = tf.sigmoid
    elif activation == "relu":
        activation_op = tf.nn.relu
    else:
        raise ValueError("{} activation not supported".format(activation))

    # Data.
    data = tfds.load('mnist', split=mode)
    images = tf.constant(data.images, dtype=tf.float32, name="MNIST_images")
    images = tf.reshape(images, [-1, 28, 28, 1])
    labels = tf.constant(data.labels, dtype=tf.int64, name="MNIST_labels")

    # Network.
    mlp = snt.nets.MLP(list(layers) + [10],
                       activation=activation_op,
                       initializers=initializers)
    network = snt.Sequential([snt.BatchFlatten(), mlp])

    def build():
        indices = tf.random_uniform([batch_size], 0, data.num_examples, tf.int64)
        batch_images = tf.gather(images, indices)
        batch_labels = tf.gather(labels, indices)
        output = network(batch_images)
        return _xent_loss(output, batch_labels)

    return build


def mnist_conv(batch_norm=True,
               batch_size=128,
               mode="train"):
    # Data.
    tfds.load('mnist', split='train')
    images = tf.constant(data.images, dtype=tf.float32, name="MNIST_images")
    images = tf.reshape(images, [-1, 28, 28, 1])
    labels = tf.constant(data.labels, dtype=tf.int64, name="MNIST_labels")

    def network(inputs, training=True):
        def _conv_activation(x):
            return tf.nn.max_pool(tf.nn.relu(x),
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding="VALID")

        def conv_layer(inputs, strides, c_h, c_w, output_channels, padding, name):
            # get size of last layer
            n_channels = int(inputs.get_shape()[-1])
            with tf.variable_scope(name) as scope:
                # init random weights
                kernel1 = tf.get_variable('weights1',
                                          shape=[c_h, c_w, n_channels, output_channels],
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(stddev=0.01)
                                          )

                # init bias
                biases1 = tf.get_variable('biases1', [output_channels], initializer=tf.constant_initializer(0.0))
            inputs = tf.nn.conv2d(inputs, kernel1, [1, strides, strides, 1], padding)
            inputs = tf.nn.bias_add(inputs, biases1)
            if batch_norm:
                inputs = tf.layers.batch_normalization(inputs, training=training)
            inputs = _conv_activation(inputs)
            return inputs

        inputs = conv_layer(inputs, 1, 3, 3, 16, "VALID", 'conv_layer1')
        inputs = conv_layer(inputs, 1, 5, 5, 32, "VALID", 'conv_layer2')
        inputs = tf.reshape(inputs, [batch_size, -1])
        fc_shape2 = int(inputs.get_shape()[1])
        # Initialize random weights and save in fc_weights
        weights = tf.get_variable("fc_weights",
                                  shape=[fc_shape2, 10],
                                  dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(stddev=0.01))
        # Initialize constant biases and save in fc_bias
        bias = tf.get_variable("fc_bias",
                               shape=[10, ],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))
        # add dense layer with weights, biases and the relu activation function
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, weights), bias))

    def build():
        # Generate batch of random indices 
        indices = tf.random_uniform([batch_size], 0, data.num_examples, tf.int64)
        # Get images and labels for the indices
        batch_images = tf.gather(images, indices)
        batch_labels = tf.gather(labels, indices)
        # create network with those images
        output = network(batch_images)
        return _xent_loss(output, batch_labels)

    return build


def nilm_seq(activation="sigmoid", sequence_length=19, mains_mean=1800, mains_std=600,
          batch_size=512, n_epochs=10, appliance_params={}, chunk_wise_training=False,
          mode="train"):
    
    models = OrderedDict()
    file_prefix = "{}-temp-weights".format(self.MODEL_NAME.lower())
    MODEL_NAME = "RNN"
    power = {'mains ': ['active '], 'appliance ': ['active ']}
    appliances = ['fridge']
    drop_nans = True
    window_size = 599 # According to seq paper
    sample_period = 1
    artificial_aggregate = False # TODO Check what it does and what is better?

    
    redd = nilmtk.DataSet('./data/redd.h5')
    mains, subs = _get_mains_and_subs(redd, appliances, power, drop_nans, artificial_aggregate)
    mains, appliances = _do_preprocessing(mains, subs, 'train', window_size)
    #TODO use tf.constant to turn mains/subs into tensors?

    def build():
        if self.sequence_length % 2 == 0:
            print("Sequence length should be odd!")
            raise SequenceLengthError
        
        indices = tf.random_uniform([batch_size], 0, len(mains), tf.int64)
        batch_mains = tf.gather(mains, indices)
        batch_appliances = tf.gather(appliances, indices)
        output = network_seq(mains) #TODO not the whole list?
        return _xent_loss(output, batch_labels)
        
    def conv_layer(inputs, strides, c_s, output_channels, padding, name, training):
        # get size of last layer
        n_channels = int(inputs.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            # init random weights
            kernel1 = tf.get_variable('weights1',
                                      shape=[c_s, n_channels, output_channels],
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(stddev=0.01)
                                      )

            # init bias
            biases1 = tf.get_variable('biases1', [output_channels], initializer=tf.constant_initializer(0.0))
        inputs = tf.nn.conv1d(inputs, kernel1, strides, padding)
        inputs = tf.nn.bias_add(inputs, biases1)
        if batch_norm:
            inputs = tf.layers.batch_normalization(inputs, training=training)# TODO necessary?
        inputs = _conv_activation(inputs) # TODO this already adds relu activation plus some more -> all required?
        return inputs
        
    def dense_layer(inputs, units, relu=True):
        # -1 means autofill
        inputs = tf.reshape(inputs, [units, -1]) # TODO what effect does this have? Is it necessary?
        fc_shape2 = int(inputs.get_shape()[1])
        # Initialize random weights and save in fc_weights
        weights = tf.get_variable("fc_weights",
                                  shape=[fc_shape2, 10],
                                  dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(stddev=0.01))
        # Initialize constant biases and save in fc_bias
        bias = tf.get_variable("fc_bias",
                               shape=[10, ],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))
        # add dense layer with weights, biases and the relu activation function
        return tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, weights), bias))
        
    # TODO rework based on paper and nilm implementation
    def network_seq(inputs, training=True):
        inputs = conv_layer(inputs, 1, 10, 30, "VALID", 'conv_layer1', training)
        inputs = conv_layer(inputs, 1, 8, 30, "VALID", 'conv_layer2', training)
        inputs = conv_layer(inputs, 1, 6, 40, "VALID", 'conv_layer3', training)
        inputs = conv_layer(inputs, 1, 5, 50, "VALID", 'conv_layer4', training)
        inputs = conv_layer(inputs, 1, 5, 50, "VALID", 'conv_layer5', training)
        inputs = dense_layer(inputs, 1024)
        inputs = dense_layer(inputs, 1) # TODO final layer should be linear? How?
        return inputs
    
    
    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
        # If no appliance wise parameters are provided, then copmute them using the first chunk
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)

        print("...............RNN partial_fit running...............")
        # Do the pre-processing, such as  windowing and normalizing
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')

        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.values.reshape((-1, self.sequence_length, 1))
        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0)
            app_df_values = app_df.values.reshape(( -1, 1 ))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        for appliance_name, power in train_appliances:
            # Check if the appliance was already trained. If not then create a new model for it
            if appliance_name not in self.models:
                print("First model training for ", appliance_name)
                self.models[appliance_name] = self.return_network()
            # Retrain the particular appliance
            else:
                print("Started Retraining model for ", appliance_name)

            model = self.models[appliance_name]
            if train_main.size > 0:
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Do validation when you have sufficient samples
                    filepath = self.file_prefix + "-{}-epoch{}.h5".format(
                            "_".join(appliance_name.split()),
                            current_epoch,
                    )
                    checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
                    model.fit(
                            train_main, power,
                            validation_split=.15,
                            epochs=self.n_epochs,
                            batch_size=self.batch_size,
                            callbacks=[ checkpoint ],
                    )
                    model.load_weights(filepath)

    def disaggregate_chunk(self,test_main_list,model=None,do_preprocessing=True):

        if model is not None:
            self.models = model

        # Preprocess the test mains such as windowing and normalizing

        if do_preprocessing:
            test_main_list = self.call_preprocessing(
                test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1, self.sequence_length, 1))
            disggregation_dict = {}
            for appliance in self.models:
                prediction = self.models[appliance].predict(test_main,batch_size=self.batch_size)
                prediction = self.appliance_params[appliance]['mean'] + prediction * self.appliance_params[appliance]['std']
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    

    def set_appliance_params(train_appliances):
        # Find the parameters using the first
        for (app_name, df_list) in train_appliances:
            l = np.array(pd.concat(df_list, axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std < 1:
                app_std = 100
            self.appliance_params.update({app_name: {'mean': app_mean, 'std': app_std}})
        print(self.appliance_params)

    return build

    # TODO use @tf.function to integrate custom functions for preprocessing etc into the tf session flow

