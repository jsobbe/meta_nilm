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

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import sonnet as snt
import tensorflow as tf
import pdb
from vgg16 import VGG16
import tensorflow_probability as tfp

from nilmtk import *
import nilm_config

tfd = tfp.distributions

_nn_initializers = {
    "w": tf.random_normal_initializer(mean=0, stddev=0.01),
    "b": tf.random_normal_initializer(mean=0, stddev=0.01),
}


def get_mains_and_subs_train(datasets, appliances, power, drop_nans, sample_period=1, artificial_aggregate=False):

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

def call_preprocessing(mains_lst, submeters_lst, method, window_size):
    mains_mean = 1800 #TODO check
    mains_std = 600
    
    if method == 'train':
        mains_df_list = []
        for mains in mains_lst:
            new_mains = mains.values.flatten()
            n = window_size
            units_to_pad = n // 2
            new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
            new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
            new_mains = (new_mains - mains_mean) / mains_std
            mains_df_list.append(pd.DataFrame(new_mains))

        appliance_list = []
        for app_index, (app_name, app_df_list) in enumerate(submeters_lst):
            appliance_params = _get_appliance_params(submeters_lst)
            if app_name in appliance_params:
                app_mean = appliance_params[app_name]['mean']
                app_std = appliance_params[app_name]['std']
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
            n = window_size
            units_to_pad = n // 2
            new_mains = np.pad(new_mains, (units_to_pad, units_to_pad), 'constant', constant_values=(0, 0))
            new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
            new_mains = (new_mains - mains_mean) / mains_std
            mains_df_list.append(pd.DataFrame(new_mains))
        return mains_df_list
    
#TODO could be optimized if it wouldnt need to be set every time
def _get_appliance_params(train_appliances):
    appliance_params = {}
    # Find the parameters using the first
    for (app_name,df_list) in train_appliances:
        l = np.array(pd.concat(df_list,axis=0))
        app_mean = np.mean(l)
        app_std = np.std(l)
        if app_std<1:
            app_std = 100
        appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})
    return appliance_params


def model(mains=None, appliances=None, mains_len=0, activation="sigmoid", 
          mains_mean=1800, mains_std=600, mode="train", load=False):
    
#     models = OrderedDict()
    file_prefix = "{}-temp-weights".format("nilm-seq")
    window_size = nilm_config.WINDOW_SIZE
    batch_size = nilm_config.BATCH_SIZE
    batch_norm = nilm_config.BATCH_NORM
    

    """
    Build the whole tf pipeline.
    """
    def build():
        if window_size % 2 == 0:
            print("Sequence length should be odd!")
            raise SequenceLengthError
        
        indices_t = tf.random_uniform([batch_size], 0, mains_len, tf.int64)
        mains_batch = tf.gather(mains, indices_t, axis = 0)
        appl_batch = tf.gather(appliances, indices_t, axis = 0)
        
        output = tf.squeeze(network_seq(mains_batch, load))
        
        return tf.losses.mean_squared_error(labels=appl_batch, predictions=output), appl_batch, output
              
    def conv_layer(inputs, strides, filter_size, output_channels, padding, name, training, load=False):
        # get size of last layer
        n_channels = int(inputs.get_shape()[-1])
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            if load:
                # init random weights
                kernel1 = tf.get_variable('weights',
                                          dtype=tf.float32,
                                          initializer=tf.constant(np.load('./nilm_models/' + name + '-weights.npy'))
                                          )

                # init bias
                biases1 = tf.get_variable('biases', 
                                          initializer=tf.constant(np.load('./nilm_models/' + name + '-biases.npy')))
            else:
                # init random weights
                kernel1 = tf.get_variable('weights',
                                          shape=[filter_size, n_channels, output_channels],
                                          dtype=tf.float32,
                                          initializer=tf.random_normal_initializer(stddev=0.01)
                                          )

                # init bias
                biases1 = tf.get_variable('biases', [output_channels], 
                                          initializer=tf.constant_initializer(0.0))
        inputs = tf.squeeze(tf.nn.conv1d(tf.expand_dims(inputs, axis=0), kernel1, strides, padding))
        inputs = tf.nn.bias_add(inputs, biases1)
        if batch_norm:
            inputs = tf.layers.batch_normalization(inputs, training=training)# TODO necessary?
        inputs = tf.nn.relu(inputs)
        return inputs
        
    def dense_layer(inputs, units, name, relu=True, load=False):
        # -1 means autofill
#         inputs = tf.reshape(inputs, [units, -1])
        fc_shape2 = int(inputs.get_shape()[-1])
        # Initialize random weights and save in fc_weights
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            if load:
                # init random weights
                weights = tf.get_variable('fc_weights',
                                          dtype=tf.float32,
                                          initializer=tf.constant(np.load('./nilm_models/' + name + '-fc_weights.npy'))
                                          )

                # init bias
                bias = tf.get_variable('fc_bias', 
                                          initializer=tf.constant(np.load('./nilm_models/' + name + '-fc_bias.npy')))
            else:
                weights = tf.get_variable("fc_weights",
                                      shape=[fc_shape2, units],
                                      dtype=tf.float32,
                                      initializer=tf.random_normal_initializer(stddev=0.01))
                # Initialize constant biases and save in fc_bias
                bias = tf.get_variable("fc_bias",
                                   shape=[units, ],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))
        # add dense layer with weights, biases and the relu activation function
        return tf.nn.bias_add(tf.matmul(inputs, weights), bias)
        
    # TODO rework based on paper and nilm implementation
    def network_seq(inputs, training=True, load=False):
        inputs = conv_layer(inputs, strides=1, filter_size=10, output_channels=30, 
                            padding="SAME", name='conv_1', training=training, load=load)
        inputs = conv_layer(inputs, strides=1, filter_size=8, output_channels=30, 
                            padding="SAME", name='conv_2', training=training, load=load)
        inputs = conv_layer(inputs, strides=1, filter_size=6, output_channels=40, 
                            padding="SAME", name='conv_3', training=training, load=load)
        inputs = conv_layer(inputs, strides=1, filter_size=5, output_channels=50, 
                            padding="SAME", name='conv_4', training=training, load=load)
        inputs = tf.nn.dropout(inputs, rate=0.2)
        inputs = conv_layer(inputs, strides=1, filter_size=5, output_channels=50, 
                            padding="SAME", name='conv_5', training=training, load=load)
#         print(str(inputs.get_shape().as_list()))
        inputs = tf.reshape(inputs, [batch_size, -1])
        inputs = tf.nn.dropout(inputs, rate=0.2)
#         inputs = tf.layers.Flatten()(inputs)
#         inputs = tf.layers.dense(inputs=inputs, units=1024, activation=tf.nn.relu)
        inputs = tf.nn.relu(dense_layer(inputs, 1024, 'dense_1', load=load)) #TODO make 1024 work
        inputs = tf.nn.dropout(inputs, rate=0.2)
#         inputs = tf.layers.dense(inputs=inputs, units=1)
        inputs = dense_layer(inputs, 1, 'dense_2', load=load) # TODO final layer should be linear? How?
        return inputs


    return build