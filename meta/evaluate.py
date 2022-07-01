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
"""Learning 2 Learn evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import pdb
import pickle
import pandas as pd
import random

from six.moves import xrange
from tensorflow.contrib.learn.python.learn import monitored_session as ms
import tensorflow as tf

import meta_dm_eval as meta_dm
import meta_rnnprop_eval as meta_rnn
import util
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import conf_eval
import conf_nilm
import nilm_seq2point
import pipeline_util

# SEEDS = random.sample(range(0, 100), conf_eval.NUM_RUNS)

def main(_):
    font = {'family' : 'normal',
        'size'   : 16}
    matplotlib.rc('font', **font)
    
    np.set_printoptions(precision=3)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    for appliance in conf_eval.APPLIANCES:
        appliance_data = {}
        try:
            appliance_data['mains'], appliance_data['appls'] = nilm_seq2point.fetch_and_preprocess_data(mode='eval', appliance=appliance)
        except KeyError:
            print('no data found for appliance ', appliance)
            continue
        print('For {} found {}/{} data entries.'.format(appliance, appliance_data['mains'].size, appliance_data['appls'].size))
        
        results = {}
        models = {}
        best_losses = {}
        for optimizer_name, data in conf_eval.OPTIMIZERS.items():
            path = data['path'] if '_' in optimizer_name else None
            shared_net = data['shared_net'] if '_' in optimizer_name else None
            result_directory = conf_eval.OUTPUT_PATH + appliance + '/'
            if not os.path.exists(result_directory):
                os.mkdir(result_directory)
            results[optimizer_name] = list()
            models[optimizer_name] = list()
            best_losses[optimizer_name] = 50 # arbitrary number higher than any expected realistic loss
            # Configuration.
            num_unrolls = conf_eval.NUM_STEPS

            # Problem, NET_CONFIG = predefined conf for META-net, NET_ASSIGNMENTS = None
            problem, mains_p, appl_p = nilm_seq2point.model(mode='eval', optimizer=optimizer_name) 
            if '_' in optimizer_name:
                net_config, net_assignments = util.get_config(conf_eval.PROBLEM, path, net_name='rnn' if 'rnn' in optimizer_name else None, shared_net=shared_net)

            step=None
            unroll_len=None

            print('\nRunning evaluation for optimizer :', optimizer_name)
            print('------------------------------------------------')

            # Optimizer setup.
            if "adam" in optimizer_name:
                cost_op, gt, pred = problem()
                problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                problem_reset = tf.variables_initializer(problem_vars)

                optimizer = tf.train.AdamOptimizer(conf_eval.LEARNING_RATE)
                optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
                update = optimizer.minimize(cost_op)
                reset = [problem_reset, optimizer_reset]
                
            elif "sgd" in optimizer_name:
                cost_op, gt, pred = problem()
                problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                problem_reset = tf.variables_initializer(problem_vars)

                optimizer = tf.train.GradientDescentOptimizer(conf_eval.LEARNING_RATE)
                optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
                update = optimizer.minimize(cost_op)
                reset = [problem_reset, optimizer_reset]

            elif "rmsprop" in optimizer_name:
                cost_op, gt, pred = problem()
                problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                problem_reset = tf.variables_initializer(problem_vars)

                optimizer = tf.train.RMSPropOptimizer(conf_eval.LEARNING_RATE)
                optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
                update = optimizer.minimize(cost_op)
                reset = [problem_reset, optimizer_reset]

            elif "adagrad" in optimizer_name:
                cost_op, gt, pred = problem()
                problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                problem_reset = tf.variables_initializer(problem_vars)

                optimizer = tf.train.AdagradOptimizer(conf_eval.LEARNING_RATE)
                optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
                update = optimizer.minimize(cost_op)
                reset = [problem_reset, optimizer_reset]
                
            elif "adadelta" in optimizer_name:
                cost_op, gt, pred = problem()
                problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                problem_reset = tf.variables_initializer(problem_vars)

                optimizer = tf.train.AdadeltaOptimizer(conf_eval.LEARNING_RATE)
                optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
                update = optimizer.minimize(cost_op)
                reset = [problem_reset, optimizer_reset]

            elif "momentum" in optimizer_name:
                cost_op, gt, pred = problem()
                problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                problem_reset = tf.variables_initializer(problem_vars)

                optimizer = tf.train.MomentumOptimizer(conf_eval.LEARNING_RATE, 0.01)
                optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
                update = optimizer.minimize(cost_op)
                reset = [problem_reset, optimizer_reset]

            elif "dm" in optimizer_name:
                if path is None:
                    print("Evaluating untrained L2L optimizer")
                optimizer = meta_dm.MetaOptimizer(**net_config)
                meta_loss, _, problem_vars, _, gt, pred = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)
                _, update, reset, cost_op, _ = meta_loss

            elif 'rnn' in optimizer_name:
                if path is None:
                    print("Evaluating untrained L2L optimizer")
                optimizer = meta_rnn.MetaOptimizer(conf_eval.BETA_1, conf_eval.BETA_2, **net_config)
                meta_loss, _, problem_vars, step, gt, pred = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)
                _, update, reset, cost_op, _ = meta_loss
                unroll_len = 1

            else:
                raise ValueError("{} is not a valid optimizer".format(optimizer_name))

            # Run evaluation multiple times
            for seed in conf_eval.SEEDS:
                tf.set_random_seed(seed)
                random.seed(seed)
                print('\nEvaluation with seed=', seed)

                with ms.MonitoredSession() as sess:
                    sess.run(reset)
                    # Prevent accidental changes to the graph.
                    tf.get_default_graph().finalize()

                    total_time = 0
                    total_cost = 0
                    loss_record = []
                    nilm_vars = []
                    for e in xrange(conf_eval.NUM_EPOCHS):
                        mains_data = appliance_data['mains']
                        appl_data = appliance_data['appls']
                        
                        # Training.
                        if 'rnn' in optimizer_name:
                            time, cost, result, nilm_vars = util.run_eval_epoch(sess, cost_op, [update], num_unrolls, step=step, unroll_len=unroll_len, feed_dict={mains_p:mains_data, appl_p:appl_data})
                        else:
                            time, cost, result, nilm_vars = util.run_eval_epoch(sess, cost_op, [update], num_unrolls, feed_dict={mains_p:mains_data, appl_p:appl_data})
                        total_time += time
                        total_cost += sum(cost) / num_unrolls
                        loss_record += cost


                    print('avg_cost:', total_cost)
                    print('loss_record:', loss_record)
                    print('final_loss:', cost[-1])
                    print('problem_vars:', len(problem_vars))
                    print('nilm_vars:', len(nilm_vars))
                    print('best:', best_losses[optimizer_name], flush=True)

                    results[optimizer_name].append(np.array(loss_record))

                    if cost[-1] < best_losses[optimizer_name]:
                        best_losses[optimizer_name] = cost[-1]
                        
                    # Save nilm model on first iteration
                    if conf_eval.SEEDS[0] == seed:
                        temp_vars = {}
#                         if not nilm_vars:
                        nilm_vars = conf_nilm.NILM_VARS if not conf_nilm.BATCH_NORM else conf_nilm.NILM_VARS_BATCH_NORM
                        for var, name in zip(problem_vars, nilm_vars):
                            run_var = sess.run(var)
                            temp_vars[name]=run_var
#                         else:
#                             for i in range(len(problem_vars)):
#                                 temp_vars[problem_vars[i].name]=nilm_vars[i]
#                             net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
#                                   scope='S2P')
#                             bn_vars = []
#                             for layer in conf_nilm.CONV_LAYERS:
#                                 bn_vars.append(sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
#                                   scope=layer + '/batch_normalization/gamma')[0]))
#                                 bn_vars.append(sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
#                                   scope=layer + '/batch_normalization/beta')[0]))
#                             nilm_vars.append(bn_vars)
#                             for var in net_vars:
#                                 run_var = sess.run(var)
#                                 print('Batch_n var: ', str(run_var))
#                                 temp_vars[var.name]=run_var
                        models[optimizer_name] = temp_vars
                        print('\nTEMP VARS:')
                        print(str(temp_vars.keys()))

                    # Results.
                    util.print_stats("Epoch {}".format(conf_eval.NUM_EPOCHS), total_cost,
                                     total_time, conf_eval.NUM_EPOCHS)

            _plot_optimizer_results(results[optimizer_name], optimizer_name, result_directory)
            _save_optimizer_results(results[optimizer_name], optimizer_name, appliance)
            pipeline_util.log_pipeline_run(mode='eval', optimizer=optimizer_name, final_loss = cost[-1], avg_loss=total_cost)

            if conf_eval.SAVE_MODEL:
                _save_optimized_nilm_model(models[optimizer_name], appliance, optimizer_name)

            tf.reset_default_graph()

        _plot_appliance_results(results, result_directory)
        
        
        for opt, best_loss in best_losses.items():
            print('Best final loss achieved for appliance ', appliance, ' by optimizer ', opt, ': ', best_loss)
        
        
def _save_optimized_nilm_model(nilm_vars, appliance_name, optimizer_name):
    directory = conf_eval.NILM_MODEL_PATH + appliance_name + '/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory += optimizer_name + '/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    for name, var in nilm_vars.items():
        name = name.replace('/','-')
        name = name.split(':')[0]
        name = name.replace('S2P-' ,'')
        if 'batch' in name:
            print(name)
            print(str(var))
        np.save(directory + name, var)
    
#     count = 0
#     for j in range(len(nilm_vars)):
#         if not 'batch_norm' in problem_vars[j].name:
#             np.save(directory + conf_nilm.NILM_VARS[count], nilm_vars[j])
#             count += 1
#         np.save(directory + conf_nilm.NILM_VARS[j], nilm_vars[j])
        
def _save_optimizer_results(results, optimizer, appliance):
    if not os.path.exists(conf_eval.OUTPUT_PATH + appliance + '/'):
        os.mkdir(conf_eval.OUTPUT_PATH + appliance + '/')
    output_file = '{}{}/{}_eval_loss_record.pickle-{}'.format(conf_eval.OUTPUT_PATH, appliance, optimizer, conf_eval.PROBLEM)
    with open(output_file, 'wb') as l_record:
        pickle.dump(results, l_record)
    print("Saving evaluate loss record {}".format(output_file))
       
def _plot_optimizer_results(results, optimizer, directory):
    average = np.mean(results, axis=0)
    error = np.std(results, axis=0)
    maxs = average + error
    mins = average - error

    plt.figure(figsize=(21, 9))
    plt.plot(average, label='Average', linewidth='2', color='blue')
    if len(conf_eval.SEEDS) > 1:
        plt.fill_between(average, mins, maxs,alpha=0.3, color='blue')
    #for r in results:
    #    plt.plot(r, linestyle='dotted', color='grey', linewidth='1')
    plt.legend()
    plt.title(optimizer)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale("log")
    plt.savefig(directory + optimizer + '.png')
    
    
    plt.figure(figsize=(21, 9))
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale("log")
    plt.plot(np.convolve(average, np.ones(10)/10, mode='valid'), label='Moving average (10)', linewidth='2')
    plt.legend()
    plt.savefig(directory + optimizer + '_avg.png')

def _plot_appliance_results(results, directory):
    plt.figure(figsize=(21, 9))
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale("log")
    for optimizer, result in results.items():
        average = np.mean(result, axis=0)
        plt.plot(average, label=optimizer, linewidth='1')
    plt.legend()
    plt.savefig(directory + 'aggregate.png')
    
    
    plt.figure(figsize=(21, 9))
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale("log")
    for optimizer, result in results.items():
        average = np.mean(result, axis=0)
        plt.plot(np.convolve(average, np.ones(10)/10, mode='valid'), label=optimizer, linewidth='2')
    plt.legend()
    plt.savefig(directory + 'aggregate_avg.png')
        

if __name__ == "__main__":
    tf.app.run()
