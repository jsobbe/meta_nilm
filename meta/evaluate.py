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

SEEDS = random.sample(range(0, 100), conf_eval.NUM_RUNS)

def main(_):
    font = {'family' : 'normal',
        'size'   : 16}
    matplotlib.rc('font', **font)
    
    np.set_printoptions(precision=3)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    appliance_data = {}
        
    for appliance in conf_eval.APPLIANCES:
        appliance_data[appliance] = {}
        appliance_data[appliance]['mains'], appliance_data[appliance]['appls'] = nilm_seq2point.preprocess_data(mode='eval', appliance=appliance)
        print('For {} found {}/{} data entries.'.format(appliance, appliance_data[appliance]['mains'].size, appliance_data[appliance]['appls'].size))
        
        results = {}
        models = {}
        best_losses = {}
        for optimizer_name, path in conf_eval.OPTIMIZERS.items():
            result_directory = conf_eval.OUTPUT_PATH + appliance + '/'
            if not os.path.exists(result_directory):
                os.mkdir(result_directory)
            results[optimizer_name] = list()
            models[optimizer_name] = list()
            best_losses[optimizer_name] = 50 # arbitrary number higher than any expected realistic loss
            # Configuration.
            num_unrolls = conf_eval.NUM_STEPS

            # Problem, NET_CONFIG = predefined conf for META-net, NET_ASSIGNMENTS = None
            problem, mains_p, appl_p = nilm_seq2point.model(mode='eval', appliance=appliance) 
            net_config, net_assignments = util.get_config(conf_eval.PROBLEM, path, net_name='rnn' if 'rnn' in optimizer_name else None)

            step=None
            unroll_len=None

            print('\nRunning evaluation for optimizer :', optimizer_name)
            print('------------------------------------------------')

            # Optimizer setup.
            if optimizer_name == "adam":
                cost_op, gt, pred = problem()
                problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                problem_reset = tf.variables_initializer(problem_vars)

                optimizer = tf.train.AdamOptimizer(conf_eval.LEARNING_RATE)
                optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
                update = optimizer.minimize(cost_op)
                reset = [problem_reset, optimizer_reset]

            elif optimizer_name == "rmsprop":
                cost_op, gt, pred = problem()
                problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                problem_reset = tf.variables_initializer(problem_vars)

                optimizer = tf.train.RMSPropOptimizer(conf_eval.LEARNING_RATE)
                optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
                update = optimizer.minimize(cost_op)
                reset = [problem_reset, optimizer_reset]

            elif optimizer_name == "adagrad":
                cost_op, gt, pred = problem()
                problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                problem_reset = tf.variables_initializer(problem_vars)

                optimizer = tf.train.AdagradOptimizer(conf_eval.LEARNING_RATE)
                optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
                update = optimizer.minimize(cost_op)
                reset = [problem_reset, optimizer_reset]
                
            elif optimizer_name == "adadelta":
                cost_op, gt, pred = problem()
                problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                problem_reset = tf.variables_initializer(problem_vars)

                optimizer = tf.train.AdadeltaOptimizer(conf_eval.LEARNING_RATE)
                optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
                update = optimizer.minimize(cost_op)
                reset = [problem_reset, optimizer_reset]

            elif optimizer_name == "momentum":
                cost_op, gt, pred = problem()
                problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                problem_reset = tf.variables_initializer(problem_vars)

                optimizer = tf.train.MomentumOptimizer(conf_eval.LEARNING_RATE, 0.01)
                optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
                update = optimizer.minimize(cost_op)
                reset = [problem_reset, optimizer_reset]

            elif optimizer_name == "dm" or optimizer_name == 'dme':
                if path is None:
                    logging.warning("Evaluating untrained L2L optimizer")
                optimizer = meta_dm.MetaOptimizer(**net_config)
                meta_loss, _, problem_vars, _, gt, pred = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)
                _, update, reset, cost_op, _ = meta_loss

            elif 'rnn' in optimizer_name:
                if path is None:
                    logging.warning("Evaluating untrained L2L optimizer")
                optimizer = meta_rnn.MetaOptimizer(conf_eval.BETA_1, conf_eval.BETA_2, **net_config)
                meta_loss, _, problem_vars, step, gt, pred = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)
                _, update, reset, cost_op, _ = meta_loss
                unroll_len = 1

            else:
                raise ValueError("{} is not a valid optimizer".format(optimizer_name))

            # Run evaluation multiple times
            for i in xrange(conf_eval.NUM_RUNS):
                tf.set_random_seed(SEEDS[i])
                print('\nEvaluation iteration #', i, ' with seed=', SEEDS[i])

                with ms.MonitoredSession() as sess:
                    sess.run(reset)
                    # Prevent accidental changes to the graph.
                    tf.get_default_graph().finalize()

                    total_time = 0
                    total_cost = 0
                    loss_record = []
                    for e in xrange(conf_eval.NUM_EPOCHS):
                        mains_data = appliance_data[appliance]['mains']
                        appl_data = appliance_data[appliance]['appls']
                        
                        # Training.
                        if 'rnn' in optimizer_name:
                            time, cost, result = util.run_eval_epoch(sess, cost_op, [update], num_unrolls, step=step, unroll_len=unroll_len, feed_dict={mains_p:mains_data, appl_p:appl_data})
                        else:
                            time, cost, result = util.run_eval_epoch(sess, cost_op, [update], num_unrolls, feed_dict={mains_p:mains_data, appl_p:appl_data})
                        total_time += time
                        total_cost += sum(cost) / num_unrolls
                        loss_record += cost


                    print('avg_cost:', total_cost)
                    print('loss_record:', loss_record)
                    print('final_loss:', cost[-1])

                    results[optimizer_name].append(np.array(loss_record))

                    if cost[-1] < best_losses[optimizer_name]:
                        best_losses[optimizer_name] = cost[-1]
                        nilm_vars=[]
                        for var in problem_vars:
                            nilm_vars.append(sess.run(var))
                        models[optimizer_name] = nilm_vars

                    # Results.
                    util.print_stats("Epoch {}".format(conf_eval.NUM_EPOCHS), total_cost,
                                     total_time, conf_eval.NUM_EPOCHS)

            _plot_optimizer_results(results[optimizer_name], optimizer_name, result_directory)
            _save_optimizer_results(results[optimizer_name], optimizer_name, appliance)
            pipeline_util.log_pipeline_run(mode='eval', optimizer=optimizer_name, final_loss = cost[-1], avg_loss=total_cost)
                    #gt_final = sess.run(gt)
                    #print('Final ground truth appliance data has mean of ' + 
                    #      str(np.mean(gt_final)) + ' and std of ' + str(np.std(gt_final)) + '.')
                   # pred_final = sess.run(pred)
                    #print('Final predicted appliance data has mean of ' + 
                    #      str(np.mean(pred_final)) + ' and std of ' + str(np.std(pred_final)) + '.')

                    #result_df = pd.DataFrame({'gt': gt_final, 'pred': pred_final})
                   # result_df.head(20)
                    #result_df.to_csv('./meta/results/adam_test.csv')

            if conf_eval.SAVE_MODEL:
                _save_optimized_nilm_model(models[optimizer_name], problem_vars, appliance, optimizer_name)

            tf.reset_default_graph()

        _plot_appliance_results(results, result_directory)
        
        
        for opt, best_loss in best_losses.items():
            print('Best final loss achieved for appliance ', appliance, ' by optimizer ', opt, ': ', best_loss)
        
        
def _save_optimized_nilm_model(nilm_vars, problem_vars, appliance_name, optimizer_name):
    #print('VAR_X: ', type(nilm_vars))
    #print('VAR_X length: ', len(nilm_vars))
    #print('VAR_X content: ', str(nilm_vars))
    directory = conf_eval.NILM_MODEL_PATH + appliance_name + '/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory += optimizer_name + '/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    count = 0
    for j in range(len(nilm_vars)):
        if not 'batch_norm' in problem_vars[j].name:
            np.save(directory + conf_nilm.NILM_VARS[count], nilm_vars[j])
            count += 1
        
        
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
    plt.plot(maxs,label='Max', linewidth='1', color='red')
    plt.plot(mins,label='Min', linewidth='1', color='green')
    #for r in results:
    #    plt.plot(r, linestyle='dotted', color='grey', linewidth='1')
    plt.legend()
    plt.title(optimizer)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale("log")
    plt.savefig(directory + optimizer + '.png')

def _plot_appliance_results(results, directory):
    plt.figure(figsize=(21, 9))
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale("log")
    for optimizer, result in results.items():
        average = np.mean(result, axis=0)
        plt.plot(average, label=optimizer, linewidth='2')
    plt.legend()
    plt.savefig(directory + 'aggregate.png')
        

if __name__ == "__main__":
    tf.app.run()
