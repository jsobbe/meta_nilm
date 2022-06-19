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
"""Learning 2 Learn training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import random
from timeit import default_timer as timer

from six.moves import xrange
from tensorflow.contrib.learn.python.learn import monitored_session as ms
import tensorflow as tf

from data_generator import data_loader
import meta_rnnprop_train as meta
import numpy as np
import util
import nilm_seq2point
import matplotlib.pyplot as plt

import pipeline_util
import conf_train


def get_optimizer_name():
    return 'rnnprop' if not conf_train.USE_CURRICULUM and not conf_train.USE_IMITATION else 'rnnprop_e'

def main(_):
    
    loss_record = []
    validation_record = []
    appliance_data = {}
    
    save_path = conf_train.SAVE_PATH + get_optimizer_name() + '/'
    
    np.set_printoptions(precision=3)
    
    if conf_train.UNROLL_LENGTH > conf_train.NUM_STEPS:
        raise ValueError('Unroll length larger than steps!')
        
    # Configuration.
    if conf_train.USE_CURRICULUM:
        num_steps = [100, 200, 500, 1000, 1500, 2000, 2500, 3000]
        num_unrolls = [int(ns / conf_train.UNROLL_LENGTH) for ns in num_steps]
        num_unrolls_eval = num_unrolls[1:]
        curriculum_idx = 0
    else:
        num_unrolls = conf_train.NUM_STEPS // conf_train.UNROLL_LENGTH

    # Output path.
    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

    for appl in conf_train.APPLIANCES:
        appliance_data[appl] = {}
        appliance_data[appl]['mains'], appliance_data[appl]['appls'] = nilm_seq2point.preprocess_data(mode='train', appliance=appl)
        print('For {} found {}/{} data entries.'.format(appl, appliance_data[appl]['mains'].size, appliance_data[appl]['appls'].size))
        
    # Problem.
    if conf_train.CONTINUE_TRAINING:
        net_config, net_assignments = util.get_config(conf_train.PROBLEM, save_path + 'rp.l2l-0', net_name='rnn')
    else:
        net_config, net_assignments = util.get_config(conf_train.PROBLEM, net_name='rnn')
    problem, mains_p, appl_p = nilm_seq2point.model(mode='train') 

    # Optimizer setup.
    optimizer = meta.MetaOptimizer(conf_train.NUM_MT, conf_train.BETA1, conf_train.BETA2, **net_config)
    minimize, scale, var_x, constants, subsets, seq_step, \
        loss_mt, steps_mt, update_mt, reset_mt, mt_labels, mt_inputs, gt, pred = optimizer.meta_minimize(
            problem, conf_train.UNROLL_LENGTH,
            learning_rate=conf_train.LEARNING_RATE,
            net_assignments=net_assignments,
            second_derivatives=conf_train.SECOND_DERIVATIVES)
    step, update, reset, cost_op, _ = minimize

    # Data generator for multi-task learning.
    if conf_train.USE_IMITATION:
        data_mt = data_loader(problem, var_x, constants, subsets, scale,
                              conf_train.MT_OPTIMIZERS, conf_train.UNROLL_LENGTH)
        if conf_train.USE_CURRICULUM:
            mt_ratios = [float(r) for r in conf_train.MT_RATIOS.split()]

    # Assign func.
    p_val_x = []
    for k in var_x:
        p_val_x.append(tf.placeholder(tf.float32, shape=k.shape))
    assign_ops = [tf.assign(var_x[k_id], p_val_x[k_id]) for k_id in range(len(p_val_x))]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with ms.MonitoredSession() as sess:
        def assign_func(val_x):
            sess.run(assign_ops, feed_dict={p: v for p, v in zip(p_val_x, val_x)})

        # tf1.14
        for rst in [reset] + reset_mt:
            sess.run(rst)

        # Start.
        start_time = timer()
        tf.get_default_graph().finalize()
        best_evaluation = float("inf")
        num_eval = 0
        improved = False
        mti = -1
        for e in xrange(conf_train.NUM_EPOCHS):
            appliance = random.choice(list(appliance_data))
            mains_data = appliance_data[appliance]['mains']
            appl_data = appliance_data[appliance]['appls']
            
            print('Run EPOCH: ', e, ' for model for ', appliance)
            # Pick a task if it's multi-task learning.
            if conf_train.USE_IMITATION:
                if conf_train.USE_CURRICULUM:
                    if curriculum_idx >= len(mt_ratios):
                        mt_ratio = mt_ratios[-1]
                    else:
                        mt_ratio = mt_ratios[curriculum_idx]
                else:
                    mt_ratio = conf_train.MT_RATIO
                if random.random() < mt_ratio:
                    mti = (mti + 1) % conf_train.NUM_MT
                    task_i = mti
                else:
                    task_i = -1
            else:
                task_i = -1

            # Training.
            if conf_train.USE_CURRICULUM:
                num_unrolls_cur = num_unrolls[curriculum_idx]
            else:
                num_unrolls_cur = num_unrolls
            
            if task_i == -1:
                time, cost = util.run_epoch(sess, cost_op, [update, step], reset,
                                            num_unrolls_cur,
                                            scale=scale,
                                            rd_scale=conf_train.USE_SCALE,
                                            rd_scale_bound=conf_train.RD_SCALE_BOUND,
                                            assign_func=assign_func,
                                            var_x=var_x,
                                            step=seq_step,
                                            unroll_len=conf_train.UNROLL_LENGTH, 
                                            feed_dict={mains_p:mains_data, appl_p:appl_data})
            else:
                data_e = data_mt.get_data(task_i, sess, num_unrolls_cur, assign_func, conf_train.RD_SCALE_BOUND,
                                        if_scale=conf_train.USE_SCALE, mt_k=conf_train.K, feed_dict={mains_p:mains_data, appl_p:appl_data})
                time, cost = util.run_epoch(sess, loss_mt[task_i], [update_mt[task_i], steps_mt[task_i]], reset_mt[task_i],
                                            num_unrolls_cur,
                                            scale=scale,
                                            rd_scale=conf_train.USE_SCALE,
                                            rd_scale_bound=conf_train.RD_SCALE_BOUND,
                                            assign_func=assign_func,
                                            var_x=var_x,
                                            step=seq_step,
                                            unroll_len=conf_train.UNROLL_LENGTH,
                                            task_i=task_i,
                                            data=data_e,
                                            label_pl=mt_labels[task_i],
                                            input_pl=mt_inputs[task_i], 
                                            feed_dict={mains_p:mains_data, appl_p:appl_data})
            print('Finished EPOCH: ', e, ' with step: ', seq_step, ' and unroll len: ', conf_train.UNROLL_LENGTH)
            loss_record.append(cost)
            print ("training_loss={}".format(cost))

            # Evaluation.
            if (e + 1) % conf_train.VALIDATION_PERIOD == 0:
                if conf_train.USE_CURRICULUM:
                    num_unrolls_eval_cur = num_unrolls_eval[curriculum_idx]
                else:
                    num_unrolls_eval_cur = num_unrolls
                num_eval += 1

                eval_cost = 0
                for _ in xrange(conf_train.VALIDATION_EPOCHS):
                    appliance = random.choice(list(appliance_data)) #TODO run for all appls?
                    mains_data = appliance_data[appliance]['mains']
                    appl_data = appliance_data[appliance]['appls']
                    
                    time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                                num_unrolls_eval_cur,
                                                step=seq_step,
                                                unroll_len=conf_train.UNROLL_LENGTH, 
                                                feed_dict={mains_p:mains_data, appl_p:appl_data})
                    eval_cost += cost

                if conf_train.USE_CURRICULUM:
                    num_steps_cur = num_steps[curriculum_idx]
                else:
                    num_steps_cur = conf_train.NUM_STEPS
                print ("epoch={}, num_steps={}, eval_loss={}".format(
                    e, num_steps_cur, eval_cost / conf_train.VALIDATION_EPOCHS), flush=True)

                if not conf_train.USE_CURRICULUM:
                    if eval_cost < best_evaluation:
                        best_evaluation = eval_cost
                        optimizer.save(sess, save_path, e + 1)
                        optimizer.save(sess, save_path, 0)
                        print ("Saving optimizer of epoch {}...".format(e + 1))
                    continue

                # Curriculum learning.
                # update curriculum
                if eval_cost < best_evaluation:
                    best_evaluation = eval_cost
                    improved = True
                    # save model
                    optimizer.save(sess, save_path, curriculum_idx)
                    optimizer.save(sess, save_path, 0)
                elif num_eval >= conf_train.MIN_NUM_EVAL and improved:
                    # restore model
                    optimizer.restore(sess, save_path, curriculum_idx)
                    num_eval = 0
                    improved = False
                    curriculum_idx += 1
                    if curriculum_idx >= len(num_unrolls):
                        curriculum_idx = -1

                    # initial evaluation for next curriculum
                    eval_cost = 0
                    for _ in xrange(conf_train.VALIDATION_EPOCHS):
                        appliance = random.choice(list(appliance_data)) #TODO run for all appls?
                        mains_data = appliance_data[appliance]['mains']
                        appl_data = appliance_data[appliance]['appls']
                        
                        time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                                    num_unrolls_eval[curriculum_idx],
                                                    step=seq_step,
                                                    unroll_len=conf_train.UNROLL_LENGTH, 
                                                    feed_dict={mains_p:mains_data, appl_p:appl_data})
                        eval_cost += cost
                    best_evaluation = eval_cost
                    print("epoch={}, num_steps={}, eval loss={}".format(
                        e, num_steps[curriculum_idx], eval_cost / conf_train.VALIDATION_EPOCHS), flush=True)
                elif num_eval >= conf_train.MIN_NUM_EVAL and not improved:
                    print ("no improve during curriculum {} --> stop".format(curriculum_idx))
                    break
                validation_record.append(eval_cost)
                    
        run_time = timer() - start_time
        
        _save_results(loss_record)
        _plot_results(loss_record)
        _plot_validation_results(validation_record)
        
        pipeline_util.log_pipeline_run(mode='train', result=loss_record, final_loss=loss_record[-1], runtime=run_time, optimizer='rnn')
        print ("total time = {}s...".format(run_time))
        
        
    
def _save_results(results):
    directory = conf_train.OUTPUT_PATH
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory += get_optimizer_name() + '/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    output_file = '{}/train_loss_record.pickle-{}'.format(directory,conf_train.PROBLEM)
    with open(output_file, 'wb') as l_record:
        pickle.dump(results, l_record)
    print("Saving evaluate loss record {}".format(output_file))
    
def _plot_results(results):
    directory = conf_train.OUTPUT_PATH
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory += get_optimizer_name() + '/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    plt.figure(figsize=(21, 9))
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale("log")
    plt.plot(results, label=get_optimizer_name(), linewidth='2')
    plt.legend()
    plt.savefig(directory + '/loss.png')
    
def _plot_validation_results(results):
    plt.figure(figsize=(10, 9))
    plt.xlabel('Validation Epochs')
    plt.ylabel('Loss')
    plt.yscale("log")
    plt.plot(results, label=get_optimizer_name(), linewidth='2')
    plt.legend()
    plt.savefig(conf_train.OUTPUT_PATH + get_optimizer_name() + '/validation_loss.png')
        

if __name__ == "__main__":
    tf.app.run()
