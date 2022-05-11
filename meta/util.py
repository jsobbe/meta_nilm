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
# limitations under the License.
# ==============================================================================
"""Learning 2 Learn utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from timeit import default_timer as timer

import numpy as np
from six.moves import xrange

import problems
import nilm_seq2point
import nilm_config
import random


def run_epoch(sess, cost_op, ops, reset, num_unrolls,
              scale=None, rd_scale=False, rd_scale_bound=3.0, assign_func=None, var_x=None,
              step=None, unroll_len=None,
              task_i=-1, data=None, label_pl=None, input_pl=None):
  print('Unrolling training epoch ', num_unrolls, ' times.')
  """Runs one optimization epoch."""
  start = timer()
  sess.run(reset)

#   v = sess.run(var_x[0])
#   print('Weights after reset: ', str(v[0]))

  cost = None
  result = ['', '']
  if task_i == -1:
      if rd_scale:
        assert scale is not None
        randomized_scale = []
        for k in scale:
          randomized_scale.append(np.exp(np.random.uniform(-rd_scale_bound, rd_scale_bound,
                            size=k.shape)))
        assert var_x is not None
        k_value_list = []
        for k_id in range(len(var_x)):
          k_value = sess.run(var_x[k_id])
          k_value = k_value / randomized_scale[k_id]
          k_value_list.append(k_value)
        assert assign_func is not None
        assign_func(k_value_list)
        # zip() creates tuples from items of both lists
        feed_rs = {p: v for p, v in zip(scale, randomized_scale)}
      else:
        feed_rs = {}
      feed_dict = feed_rs
      for i in xrange(num_unrolls):
        if step is not None:
            feed_dict[step] = i*unroll_len+1
        result = sess.run([cost_op] + ops, feed_dict=feed_dict)
        cost = result[0]
  else: # if multitask learning. But how is task selected?
      assert data is not None
      assert input_pl is not None
      assert label_pl is not None
      feed_dict = {}
      for ri in xrange(num_unrolls):
          for pl, dat in zip(label_pl, data["labels"][ri]):
              feed_dict[pl] = dat
          for pl, dat in zip(input_pl, data["inputs"][ri]):
              feed_dict[pl] = dat
          if step is not None:
              feed_dict[step] = ri * unroll_len + 1
          result = sess.run([cost_op] + ops, feed_dict=feed_dict)
          cost = result[0]
            
#   v = sess.run(var_x[0])
#   print('Weights at end of epoch: ', str(v[0]))
    
  return timer() - start, cost


def run_eval_epoch(sess, cost_op, ops, num_unrolls, step=None, unroll_len=None):
  """Runs one optimization epoch."""
  print('Unrolling evaluation epoch ', num_unrolls, ' times.')
  start = timer()
  # sess.run(reset)
  total_cost = []
  feed_dict = {}
  for i in xrange(num_unrolls):
    if step is not None:
        feed_dict[step] = i * unroll_len + 1
    result = sess.run([cost_op] + ops, feed_dict=feed_dict)
    cost = result[0]
    total_cost.append(cost)
  return timer() - start, total_cost, result[-1]


def print_stats(header, total_error, total_time, n):
  """Prints experiment statistics."""
  print(header)
  print("Log Mean Final Error: {:.2f}".format(np.log10(total_error / n)))
  print("Mean epoch time: {:.2f} s".format(total_time / n))


def _prepare_nilm_data(mode="train"):
    if mode is "train":
        data=nilm_config.DATASETS_TRAIN
    else:
        data=nilm_config.DATASETS_EVAL
    appliances = nilm_config.APPLIANCES
    window_size = nilm_config.WINDOW_SIZE
    batch_size = nilm_config.BATCH_SIZE
    do_preprocessing = nilm_config.PREPROCESSING
    load = False
    
    mains, subs = nilm_seq2point.get_mains_and_subs_train(
        data, appliances[0])#TODO

    mains, appls = nilm_seq2point.call_preprocessing(mains, subs, 'train', window_size)
    # TODO check method='train'
    
    # mains is currently list of df with many windows
    # Convert list of dataframes to a single tensor
    main_tensors = []
    mains_len = 0
    for main_df in mains:
        if not main_df.empty:
            mains_len += len(main_df)
        main_tensors.append(tf.convert_to_tensor(main_df))
    if mains_len <= 1:
        raise ValueError('No mains data found in provided time frame') 
    print('num of mains:', mains_len)

    mains_t = tf.squeeze(tf.convert_to_tensor(main_tensors))

    appl_tensors = []
    for appl_df in appls:
        appl_tensors.append(tf.convert_to_tensor(appl_df)) # TODO for more appliances
    appl_t = tf.squeeze(tf.convert_to_tensor(appl_tensors))
    return mains_t, appl_t, mains_len


def _get_default_net_config(path, net_name):
    if net_name == "rnn":
        return {
              "net": "RNNprop",
              "net_options": {
                  "layers": (20, 20),
                  "preprocess_name": "fc",
                  "preprocess_options": {"dim": 20},
                  "scale": 0.01,
                  "tanh_output": True
              },
              "net_path": path
          }
    else:
        return {
            "net": "CoordinateWiseDeepLSTM",
            "net_options": {
                "layers": (20, 20),
                "preprocess_name": "LogAndSign",
                "preprocess_options": {"k": 5},
                "scale": 0.01,
            },
            "net_path": path
        }

def _get_default_net(path, net_name):
    net_config = {
        "rp" if net_name == "rnn" else "cw": _get_default_net_config(path, net_name)
    }
    return net_config, None

def _get_net_per_layer_type(path, net_name):
    c_path = None
    d_path = None
    if path:
        c_path = path[0]
        d_path = path[1]
    net_config = {
        "conv": _get_default_net_config(c_path, net_name),
        "fc": _get_default_net_config(d_path, net_name)
    }
    conv_vars = ["conv_{}/weights".format(i) for i in xrange(1, 4)]
    conv_vars += ["conv_{}/biases".format(i) for i in xrange(1, 4)]
    #conv_vars += ["conv_batch_norm_{}/beta".format(i) for i in xrange(5)]
    fc_vars = ["dense_{}/weights".format(i) for i in xrange(1, 3)]
    fc_vars += ["dense_{}/biases".format(i) for i in xrange(1, 3)]
    #fc_vars += ["mlp/batch_norm/beta"]
    net_assignments = [("conv", conv_vars), ("fc", fc_vars)]
    return net_config, net_assignments
    


def get_config(problem_name, path=None, mode=None, net_name=None):
  """Returns problem configuration."""
  shared_net = True if net_name == 'rnn' else nilm_config.SHARED_NET
  print('Load config for path ', path, ', net name ', net_name)
# ----------------------- RELEVANT -------------------------
  if problem_name == "nilm_seq": 
    
    if mode is None:
        mode = "train" if path is None else "test"
    mains, appls, mains_len = _prepare_nilm_data(mode)
    
    problem = nilm_seq2point.model(mode=mode, mains=mains, appliances=appls, mains_len=mains_len, appliance_name='fridge') # TODO get from somewhere else
    
    if shared_net:
        net_config, net_assignments = _get_default_net(path, net_name)
    else:
        net_config, net_assignments = _get_net_per_layer_type(path, net_name)
        
# ----------------------- RELEVANT -------------------------

  
  else:
    raise ValueError("{} is not a valid problem".format(problem_name))

  return problem, net_config, net_assignments
