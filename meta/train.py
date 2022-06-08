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
import matplotlib.pyplot as plt
import matplotlib

from data_generator import data_loader
import meta_dm_train as meta_dm
import meta_rnnprop_train as meta_rnn
import numpy as np
import util

import conf_train
import conf_nilm
import nilm_seq2point



def main(_):
    font = {'family' : 'normal',
        'size'   : 16}
    matplotlib.rc('font', **font)
    
    use_curriculum = False
    use_imitation = False
    
    np.set_printoptions(precision=3)
    
    for appliance in conf_train.APPLIANCES:
        loss_records = {}
        validation_records = {}
        for optimizer_name in conf_train.OPTIMIZERS:
            loss_records[optimizer_name] = list()
            validation_records[optimizer_name] = list()
            save_path = conf_train.SAVE_PATH + optimizer_name + '/'
            
            seq_step = None
            unroll_len = None
            if 'rnn' in optimizer_name:
                unroll_len = conf_train.UNROLL_LENGTH
            if '_e' in optimizer_name:
                use_curriculum = True
                use_imitation = True
    
            if conf_train.UNROLL_LENGTH > conf_train.NUM_STEPS:
                raise ValueError('Unroll length larger than steps!')

            # Configuration.
            if use_curriculum:
                num_steps = [100, 200, 500, 1000, 1500, 2000, 2500, 3000]
                num_unrolls = [int(ns / conf_train.UNROLL_LENGTH) for ns in num_steps]
                print('Resulting unrolls for curriculum: ', num_unrolls)
                num_unrolls_eval = num_unrolls[1:]
                curriculum_idx = 0
            else:
                num_unrolls = conf_train.NUM_STEPS // conf_train.UNROLL_LENGTH

            # Output path.
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

             # Problem, NET_CONFIG = predefined conf for META-net, NET_ASSIGNMENTS = None
            net_config, net_assignments = util.get_config(conf_train.PROBLEM, net_name='rnn' if 'rnn' in optimizer_name else None)
            mains, appls = nilm_seq2point.preprocess_data(mode='train', appliance=appliance)
            problem, mains_p, appl_p = nilm_seq2point.model(mode='train', appliance=appliance) 

            # Optimizer setup.
            if 'rnn' in optimizer_name:
                optimizer = meta_rnn.MetaOptimizer(conf_train.NUM_MT, conf_train.BETA1, conf_train.BETA2, **net_config)
                minimize, scale, var_x, constants, subsets, seq_step, \
                    loss_mt, steps_mt, update_mt, reset_mt, mt_labels, mt_inputs, gt, pred = optimizer.meta_minimize(
                        problem, conf_train.UNROLL_LENGTH,
                        learning_rate=conf_train.LEARNING_RATE,
                        net_assignments=net_assignments,
                        second_derivatives=conf_train.SECOND_DERIVATIVES)
                step, update, reset, cost_op, _ = minimize 
            else:
                optimizer = meta_dm.MetaOptimizer(conf_train.NUM_MT, **net_config)
                minimize, scale, var_x, constants, subsets, \
                    loss_mt, steps_mt, update_mt, reset_mt, mt_labels, mt_inputs, gt, pred = optimizer.meta_minimize(
                        problem, conf_train.UNROLL_LENGTH,
                        learning_rate=conf_train.LEARNING_RATE,
                        net_assignments=net_assignments,
                        second_derivatives=conf_train.SECOND_DERIVATIVES)
            step, update, reset, cost_op, _ = minimize 

            # Data generator for multi-task learning.
            if use_imitation:
                data_mt = data_loader(problem, var_x, constants, subsets, scale,
                                      conf_train.MT_OPTIMIZERS, conf_train.UNROLL_LENGTH)
                if use_curriculum:
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

                #if conf_train.CONTINUE_TRAINING:
                #    optimizer.restore(sess, save_path, 0)

                # Start.
                start_time = timer()
                tf.get_default_graph().finalize()
                best_evaluation = float("inf")
                num_eval = 0
                improved = False
                mti = -1
                for e in xrange(conf_train.NUM_EPOCHS):
                    print('Run EPOCH: ', e)
                    # Pick a task if it's multi-task learning.
                    if use_imitation:
                        if use_curriculum:
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
                    if use_curriculum:
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
                                                    unroll_len=unroll_len, 
                                                    feed_dict={mains_p:mains, appl_p:appls})
                    else:
                        data_e = data_mt.get_data(task_i, sess, num_unrolls_cur, assign_func, conf_train.RD_SCALE_BOUND,
                                                if_scale=conf_train.USE_SCALE, mt_k=conf_train.K)
                        time, cost = util.run_epoch(sess, loss_mt[task_i], [update_mt[task_i], steps_mt[task_i]], reset_mt[task_i],
                                                    num_unrolls_cur,
                                                    scale=scale,
                                                    rd_scale=conf_train.USE_SCALE,
                                                    rd_scale_bound=conf_train.RD_SCALE_BOUND,
                                                    assign_func=assign_func,
                                                    var_x=var_x,
                                                    step=seq_step,
                                                    unroll_len=unroll_len,
                                                    task_i=task_i,
                                                    data=data_e,
                                                    label_pl=mt_labels[task_i],
                                                    input_pl=mt_inputs[task_i],
                                                    feed_dict={mains_p:mains, appl_p:appls})
                    print('Finished EPOCH: ', e, ' with step: ', seq_step, ' and unroll len: ', unroll_len)
                    print ("training_loss={}".format(cost))
                    loss_records[optimizer_name].append(cost)

                    # Evaluation after conf_train.VALID_PERIOD epochs.
                    if (e+1) % conf_train.VALIDATION_PERIOD == 0:
                        if use_curriculum:
                            num_unrolls_eval_cur = num_unrolls_eval[curriculum_idx]
                        else:
                            num_unrolls_eval_cur = num_unrolls
                        num_eval += 1

                        eval_cost = 0
                        for _ in xrange(conf_train.VALIDATION_EPOCHS):
                            time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                            num_unrolls_eval_cur,
                                            step=seq_step,
                                            unroll_len=conf_train.UNROLL_LENGTH,
                                            feed_dict={mains_p:mains, appl_p:appls})
                            eval_cost += cost
                        validation_records[optimizer_name].append(eval_cost)

                        if use_curriculum:
                            num_steps_cur = num_steps[curriculum_idx]
                        else:
                            num_steps_cur = conf_train.NUM_STEPS
                        print ("epoch={}, num_steps={}, eval_loss={}".format(
                            e, num_steps_cur, eval_cost / conf_train.VALIDATION_EPOCHS), flush=True)

                        if not use_curriculum:
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
                                time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                                num_unrolls_eval_cur,
                                                step=seq_step,
                                                unroll_len=conf_train.UNROLL_LENGTH)
                                eval_cost += cost
                            best_evaluation = eval_cost
                            print("epoch={}, num_steps={}, eval loss={}".format(
                                e, num_steps[curriculum_idx], eval_cost / conf_train.VALIDATION_EPOCHS), flush=True)
                        elif num_eval >= conf_train.MIN_NUM_EVAL and not improved:
                            print("no improve during curriculum {} --> stop".format(curriculum_idx))
                            break

                    #gt_final = sess.run(gt)
                    #print('Final ground truth appliance data (length=', gt_final.size, ') has mean of ' + 
                    #      str(np.mean(gt_final)) + ' and std of ' + str(np.std(gt_final)) + '.')
                    #pred_final = sess.run(pred)
                    #print('Final predicted appliance data (length=', pred_final.size, ') has mean of ' + 
                    #      str(np.mean(pred_final)) + ' and std of ' + str(np.std(pred_final)) + '.')
                
                _save_optimizer_results(loss_records[optimizer_name], appliance, optimizer_name)
        _plot_appliance_results(loss_records, appliance, optimizer_name,)
        _plot_validation_results(validation_records, appliance, optimizer_name,)

    
def _save_optimizer_results(results, appliance, optimizer):
    directory = conf_train.SAVE_PATH + appliance + '/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory += optimizer + '/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    output_file = '{}/train_loss_record.pickle-{}'.format(directory,conf_train.PROBLEM)
    with open(output_file, 'wb') as l_record:
        pickle.dump(results, l_record)
    print("Saving evaluate loss record {}".format(output_file))
    
def _plot_appliance_results(results, appliance, optimizer):
    directory = conf_train.OUTPUT_PATH + appliance + '/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory += optimizer + '/'
    if not os.path.exists(directory):
        os.mkdir(directory)
    
    plt.figure(figsize=(21, 9))
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale("log")
    for optimizer, result in results.items():
        plt.plot(result, label=optimizer, linewidth='2')
    plt.legend()
    plt.savefig(directory + '/loss.png')
    
def _plot_validation_results(results, appliance, optimizer):
    plt.figure(figsize=(10, 9))
    plt.xlabel('Validation Epochs')
    plt.ylabel('Loss')
    plt.yscale("log")
    for optimizer, result in results.items():
        plt.plot(result, label=optimizer, linewidth='2')
    plt.legend()
    plt.savefig(conf_train.OUTPUT_PATH + appliance + '/' + optimizer + '/validation_loss.png')

    
if __name__ == "__main__":
    tf.app.run()
    