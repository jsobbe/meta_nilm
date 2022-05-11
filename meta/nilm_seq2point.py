# TODO check imports!!
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import tarfile
import sys


from collections import OrderedDict

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

from nilmtk import *

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
import tensorflow as tf
import pdb
from vgg16 import VGG16
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

import nilm_config

tfd = tfp.distributions

_nn_initializers = {
    "w": tf.random_normal_initializer(mean=0, stddev=0.01),
    "b": tf.random_normal_initializer(mean=0, stddev=0.01),
}


def get_mains_and_subs_train(datasets, appliance_name):
    power = nilm_config.POWER
    sample_period = nilm_config.SAMPLE_PERIOD
    drop_nans = nilm_config.DROP_NANS
    artificial_aggregate = nilm_config.ARTIFICIAL_AGGREGATE
    # This function has a few issues, which should be addressed soon
    print("............... Loading Data for training ...................")
    # store the train_main readings for all buildings
    train_mains = []
    train_submeters = []
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
            appliance_df = next(train.buildings[building].elec[appliance_name].load(
                physical_quantity='power', ac_type=power['appliance'], sample_period=sample_period))
            appliance_df = appliance_df[[list(appliance_df.columns)[0]]] # TODO support for multiple appliances?

            if drop_nans:
                train_df, appliance_df = _dropna(train_df, appliance_df)

            if artificial_aggregate: # TODO does this make sense for only one appliance?
                print ("Creating an Artificial Aggregate")
                train_df = pd.DataFrame(np.zeros(appliance_df.shape),index =
                                        appliance_df.index,columns=appliance_df.columns)
                train_df+=app_reading

            train_mains.append(train_df)
            train_submeters.append(appliance_df)

    return train_mains,train_submeters

def _dropna(mains_df, appliance_df):
        """
        Drops the missing values in the Mains reading and appliance readings and returns consistent data by copmuting the intersection
        """
        print ("Dropping missing values")

        # The below steps are for making sure that data is consistent by doing intersection across appliances
        mains_df = mains_df.dropna()
        ix = mains_df.index
        mains_df = mains_df.loc[ix]
        appliance_df = appliance_df.dropna()
    
        ix = ix.intersection(appliance_df.index)
        mains_df = mains_df.loc[ix]
        new_appliances_df = appliance_df.loc[ix]
        return mains_df, new_appliances_df

def call_preprocessing(mains_lst, submeters_lst, method, window_size):
    print("\n< BEGIN PREPROCESSING >")
    mains_mean, mains_std = _get_mean_and_std(mains_lst)
    print('Mean in mains data: ', mains_mean)
    print('Std in mains data: ', mains_std)
    print('Max in mains data: ', np.max(np.array(pd.concat(mains_lst,axis=0))))
    print('Min in mains data: ', np.min(np.array(pd.concat(mains_lst,axis=0))))
    
    mains_df_list = []
    for mains in mains_lst:
        new_mains = mains.values.flatten()
        n = window_size
        units_to_pad = n // 2
        new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
        new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
#         print(str(new_mains))
        new_mains = (new_mains - mains_mean) / mains_std
        print('Mains mean after preprocessing: ', np.mean(new_mains))
        print('Mains std after preprocessing: ', np.std(new_mains))
        print('Mains max after preprocessing: ', np.max(new_mains))
        print('Mains min after preprocessing: ', np.min(new_mains), ' \n')
        mains_df_list.append(pd.DataFrame(new_mains))
        
    appliance_list = []
    if submeters_lst:
        # Appliances don't need to be processed, when evaluating the nilm model
        app_mean, app_std = _get_mean_and_std(submeters_lst)
        print('Mean in apps data: ', app_mean)
        print('Std in apps data: ', app_std)
        print('Max in apps data: ', np.max(np.array(pd.concat(submeters_lst,axis=0))))
        print('Min in apps data: ', np.min(np.array(pd.concat(submeters_lst,axis=0))))
        for app_df in submeters_lst:
            new_app_readings = app_df.values.reshape((-1, 1))
            # This is for choosing windows
            new_app_readings = (new_app_readings - app_mean) / app_std
            # Return as a list of dataframe
            appliance_list.append(pd.DataFrame(new_app_readings))
            print('Apps mean data after preprocessing: ', np.mean(new_app_readings))
            print('Apps std data after preprocessing: ', np.std(new_app_readings))
            print('Apps max data after preprocessing: ', np.max(new_app_readings))
            print('Apps min data after preprocessing: ', np.min(new_app_readings))
    print("< END PREPROCESSING > \n")
    
    return mains_df_list, appliance_list

    
def _get_mean_and_std(mains):
    l = np.array(pd.concat(mains,axis=0))
    mean = np.mean(l)
    std = np.std(l)
    if std<1:
        std = 100
    return mean, std


def model(mains=None, appliances=None, appliance_name='default', mains_len=0, optimizer="L2L", mode="train", load=False, batch_size=nilm_config.BATCH_SIZE):
    
    file_prefix = "{}-temp-weights".format("nilm-seq")
    window_size = nilm_config.WINDOW_SIZE
    batch_norm = nilm_config.BATCH_NORM
     

    """
    Build the whole tf pipeline.
    """
    def build():
        if window_size % 2 == 0:
            print("Sequence length should be odd!")
            raise SequenceLengthError
        if not load:
            print('Building model for optimizer ', optimizer, ' and mode ', mode)
        else:
            print('Loading model for optimizer ', optimizer, ' and mode ', mode)
        
#         with tf.Session() as sess:
#             print('Mains data has mean of ' + np.mean(mains.eval(sess)) + ' and std of ' + np.std(mains.eval(sess)) + '.')
#             print('Appliance data has mean of ' +  np.mean(appliances.eval(sess)) + ' and std of ' + np.std(appliances.eval(sess)) + '.')
         #TODO return indices too?
        
        # If no appliances are provided, model is presumably used for prediction, so only return output
        if appliances is not None:
            indices_t = tf.random_uniform([batch_size], 0, mains_len, tf.int64)
            mains_batch = tf.gather(mains, indices_t, axis = 0)
            print('Shape after gather: ', mains_batch.get_shape())
            output = tf.squeeze(network_seq(mains_batch))
            appl_batch = tf.gather(appliances, indices_t, axis = 0)
            return _mse(targets=appl_batch, outputs=output), appl_batch, output
        else:
            indices_t = tf.range(mains_len, dtype=tf.int64)
            mains_batch = tf.gather(mains, indices_t, axis = 0)
            output = tf.squeeze(network_seq(mains_batch))
            return output, mains_batch
        
    def conv_layer(inputs, strides, filter_size, output_channels, padding, name):
        # get size of last layer
        n_channels = int(inputs.get_shape()[-1])
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            if load:
                # init random weights
                kernel1 = tf.Variable(name='weights',
                                          dtype=tf.float32,
                                          initial_value=tf.constant(np.load('./nilm_models/eval/'
                                                                          + optimizer + '/'
                                                                          + name + '-weights.npy')))
                # init bias
                biases1 = tf.Variable(name='biases', 
                                           dtype=tf.float32,
                                          initial_value=tf.constant(np.load('./nilm_models/eval/'
                                                                          + optimizer + '/'
                                                                          + name + '-biases.npy')))
            else:
                # init random weights
                kernel1 = tf.get_variable('weights',
                                          shape=[filter_size, n_channels, output_channels],
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(stddev=0.01))
                print('Weights for ', name, ': ', str(kernel1))
                # init bias
                biases1 = tf.get_variable('biases', [output_channels], 
                                           dtype=tf.float32,
                                          initializer=tf.constant_initializer(0.0))
            inputs = tf.nn.conv1d(inputs, kernel1, strides, padding)
            inputs = tf.reshape(inputs, [batch_size, -1, output_channels])
            inputs = tf.nn.bias_add(inputs, biases1)
            if batch_norm:
                inputs = tf.layers.batch_normalization(inputs, training=mode=='train')# TODO necessary?
        inputs = tf.nn.relu(inputs)
        return inputs
        
    def dense_layer(inputs, units, name, relu=False):
#         inputs = tf.reshape(inputs, [units, -1])
        fc_shape2 = int(inputs.get_shape()[-1])
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            if load:
                # init random weights
                weights = tf.Variable(name='weights',
                                          dtype=tf.float32,
                                          initial_value=tf.constant(np.load('./nilm_models/eval/'
                                                                          + optimizer + '/'
                                                                          + name + '-weights.npy')))
                # init bias
                bias = tf.Variable(name='biases', 
                                       dtype=tf.float32,
                                       initial_value=tf.constant(np.load('./nilm_models/eval/' 
                                                                          + optimizer + '/'
                                                                          + name + '-biases.npy')))
            else:
                weights = tf.get_variable("weights",
                                      shape=[fc_shape2, units],
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(stddev=0.01))
                print('Weights for ', name, ': ', str(weights))
                # Initialize constant biases and save in fc_bias
                bias = tf.get_variable("biases",
                                   shape=[units],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
        # add dense layer with weights, biases and the relu activation function
        if relu:
            inputs = tf.nn.relu(tf.nn.bias_add(tf.matmul(inputs, weights), bias))
        else:
            inputs = tf.nn.bias_add(tf.matmul(inputs, weights), bias)
        return inputs
        
    # TODO rework based on paper and nilm implementation
    def network_seq(inputs):
        inputs = tf.reshape(inputs, [batch_size, window_size, -1])
        inputs = conv_layer(inputs, strides=3, filter_size=10, output_channels=30, 
                            padding="VALID", name='conv_1')
        inputs = conv_layer(inputs, strides=3, filter_size=8, output_channels=30, 
                            padding="VALID", name='conv_2')
        inputs = conv_layer(inputs, strides=2, filter_size=6, output_channels=40, 
                            padding="VALID", name='conv_3')
        inputs = conv_layer(inputs, strides=2, filter_size=5, output_channels=50, 
                            padding="VALID", name='conv_4')
        inputs = tf.nn.dropout(inputs, rate=0.2)
        inputs = conv_layer(inputs, strides=1, filter_size=5, output_channels=50, 
                            padding="VALID", name='conv_5')
        inputs = tf.nn.dropout(inputs, rate=0.2)
        inputs = tf.reshape(inputs, [batch_size, -1])
        inputs = dense_layer(inputs, 512, 'dense_1', relu=True)
        inputs = tf.nn.dropout(inputs, rate=0.2)
        inputs = dense_layer(inputs, 1, 'dense_2')
        #print('Shape post dense: ', inputs.get_shape(), ' \n')
        return inputs
    
    def _mae(targets, outputs):
        return tf.reduce_mean(tf.abs(targets - outputs))
        
    def _rmse(targets, outputs):
        return tf.sqrt(tf.reduce_mean((targets - outputs)**2))
    
    def _mse(targets, outputs):
        return tf.reduce_mean((targets - outputs)**2)
    
    def _xent_loss(outputs, targets):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs,
                                                        labels=targets)
        return tf.reduce_mean(loss)



    return build