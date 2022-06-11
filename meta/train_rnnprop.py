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

import pipeline_util

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("save_path", None, "Path for saved meta-optimizer.")

flags.DEFINE_integer("num_epochs", 200, "Number of training epochs.")
flags.DEFINE_integer("evaluation_period", 50, "Evaluation period.")
flags.DEFINE_integer("evaluation_epochs", 5, "Number of evaluation epochs.")
flags.DEFINE_integer("num_steps", 150, "Number of optimization steps per epoch.")
flags.DEFINE_integer("unroll_length", 20, "Meta-optimizer unroll length.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_boolean("second_derivatives", False, "Use second derivatives.")
flags.DEFINE_boolean("save", False, "Whether to save the resulting nilm-model.")
flags.DEFINE_boolean("load", False, "Whether to continue training saved model.") 

flags.DEFINE_float("beta1", 0.95, "")
flags.DEFINE_float("beta2", 0.95, "")

flags.DEFINE_string("problem", "nilm_seq", "Type of problem.")

flags.DEFINE_boolean("if_scale", False, "")
flags.DEFINE_float("rd_scale_bound", 3.0, "Bound for random scaling on the main optimizee.")

flags.DEFINE_boolean("if_cl", False, "")
flags.DEFINE_integer("min_num_eval", 3, "")

flags.DEFINE_boolean("if_mt", False, "")
flags.DEFINE_integer("num_mt", 1, "")
flags.DEFINE_string("optimizers", "adam", ".")
flags.DEFINE_float("mt_ratio", 0.3, "")
flags.DEFINE_string("mt_ratios", "0.3 0.3 0.3", "")
flags.DEFINE_integer("k", 1, "")


def main(_):
    np.set_printoptions(precision=3)
    
    if FLAGS.unroll_length > FLAGS.num_steps:
        raise ValueError('Unroll length larger than steps!')
        
    # Configuration.
    if FLAGS.if_cl:
        num_steps = [100, 200, 500, 1000, 1500, 2000, 2500, 3000]
        num_unrolls = [int(ns / FLAGS.unroll_length) for ns in num_steps]
        num_unrolls_eval = num_unrolls[1:]
        curriculum_idx = 0
    else:
        num_unrolls = FLAGS.num_steps // FLAGS.unroll_length

    # Output path.
    if FLAGS.save_path is not None:
        if not os.path.exists(FLAGS.save_path):
            os.mkdir(FLAGS.save_path)

    # Problem.
    net_config, net_assignments = util.get_config(FLAGS.problem, net_name='rnn')
    mains, appls = nilm_seq2point.preprocess_data(mode='train', appliance='fridge')
    problem, mains_p, appl_p = nilm_seq2point.model(mode='train', appliance='fridge') 

    # Optimizer setup.
    optimizer = meta.MetaOptimizer(FLAGS.num_mt, FLAGS.beta1, FLAGS.beta2, **net_config)
    minimize, scale, var_x, constants, subsets, seq_step, \
        loss_mt, steps_mt, update_mt, reset_mt, mt_labels, mt_inputs, gt, pred = optimizer.meta_minimize(
            problem, FLAGS.unroll_length,
            learning_rate=FLAGS.learning_rate,
            net_assignments=net_assignments,
            second_derivatives=FLAGS.second_derivatives)
    step, update, reset, cost_op, _ = minimize

    # Data generator for multi-task learning.
    if FLAGS.if_mt:
        data_mt = data_loader(problem, var_x, constants, subsets, scale,
                              FLAGS.optimizers, FLAGS.unroll_length)
        if FLAGS.if_cl:
            mt_ratios = [float(r) for r in FLAGS.mt_ratios.split()]

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
        for e in xrange(FLAGS.num_epochs):
            # Pick a task if it's multi-task learning.
            if FLAGS.if_mt:
                if FLAGS.if_cl:
                    if curriculum_idx >= len(mt_ratios):
                        mt_ratio = mt_ratios[-1]
                    else:
                        mt_ratio = mt_ratios[curriculum_idx]
                else:
                    mt_ratio = FLAGS.mt_ratio
                if random.random() < mt_ratio:
                    mti = (mti + 1) % FLAGS.num_mt
                    task_i = mti
                else:
                    task_i = -1
            else:
                task_i = -1

            # Training.
            if FLAGS.if_cl:
                num_unrolls_cur = num_unrolls[curriculum_idx]
            else:
                num_unrolls_cur = num_unrolls

            if task_i == -1:
                time, cost = util.run_epoch(sess, cost_op, [update, step], reset,
                                            num_unrolls_cur,
                                            scale=scale,
                                            rd_scale=FLAGS.if_scale,
                                            rd_scale_bound=FLAGS.rd_scale_bound,
                                            assign_func=assign_func,
                                            var_x=var_x,
                                            step=seq_step,
                                            unroll_len=FLAGS.unroll_length, 
                                            feed_dict={mains_p:mains, appl_p:appls})
            else:
                data_e = data_mt.get_data(task_i, sess, num_unrolls_cur, assign_func, FLAGS.rd_scale_bound,
                                        if_scale=FLAGS.if_scale, mt_k=FLAGS.k)
                time, cost = util.run_epoch(sess, loss_mt[task_i], [update_mt[task_i], steps_mt[task_i]], reset_mt[task_i],
                                            num_unrolls_cur,
                                            scale=scale,
                                            rd_scale=FLAGS.if_scale,
                                            rd_scale_bound=FLAGS.rd_scale_bound,
                                            assign_func=assign_func,
                                            var_x=var_x,
                                            step=seq_step,
                                            unroll_len=FLAGS.unroll_length,
                                            task_i=task_i,
                                            data=data_e,
                                            label_pl=mt_labels[task_i],
                                            input_pl=mt_inputs[task_i], 
                                            feed_dict={mains_p:mains, appl_p:appls})
            print('Finished EPOCH: ', e, ' with step: ', seq_step, ' and unroll len: ', FLAGS.unroll_length)
            print ("training_loss={}".format(cost))

            # Evaluation.
            if (e + 1) % FLAGS.evaluation_period == 0:
                if FLAGS.if_cl:
                    num_unrolls_eval_cur = num_unrolls_eval[curriculum_idx]
                else:
                    num_unrolls_eval_cur = num_unrolls
                num_eval += 1

                eval_cost = 0
                for _ in xrange(FLAGS.evaluation_epochs):
                    time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                                num_unrolls_eval_cur,
                                                step=seq_step,
                                                unroll_len=FLAGS.unroll_length, 
                                                feed_dict={mains_p:mains, appl_p:appls})
                    eval_cost += cost

                if FLAGS.if_cl:
                    num_steps_cur = num_steps[curriculum_idx]
                else:
                    num_steps_cur = FLAGS.num_steps
                print ("epoch={}, num_steps={}, eval_loss={}".format(
                    e, num_steps_cur, eval_cost / FLAGS.evaluation_epochs), flush=True)

                if not FLAGS.if_cl:
                    if eval_cost < best_evaluation:
                        best_evaluation = eval_cost
                        optimizer.save(sess, FLAGS.save_path, e + 1)
                        optimizer.save(sess, FLAGS.save_path, 0)
                        print ("Saving optimizer of epoch {}...".format(e + 1))
                    continue

                # Curriculum learning.
                # update curriculum
                if eval_cost < best_evaluation:
                    best_evaluation = eval_cost
                    improved = True
                    # save model
                    optimizer.save(sess, FLAGS.save_path, curriculum_idx)
                    optimizer.save(sess, FLAGS.save_path, 0)
                elif num_eval >= FLAGS.min_num_eval and improved:
                    # restore model
                    optimizer.restore(sess, FLAGS.save_path, curriculum_idx)
                    num_eval = 0
                    improved = False
                    curriculum_idx += 1
                    if curriculum_idx >= len(num_unrolls):
                        curriculum_idx = -1

                    # initial evaluation for next curriculum
                    eval_cost = 0
                    for _ in xrange(FLAGS.evaluation_epochs):
                        time, cost = util.run_epoch(sess, cost_op, [update], reset,
                                                    num_unrolls_eval[curriculum_idx],
                                                    step=seq_step,
                                                    unroll_len=FLAGS.unroll_length)
                        eval_cost += cost
                    best_evaluation = eval_cost
                    print("epoch={}, num_steps={}, eval loss={}".format(
                        e, num_steps[curriculum_idx], eval_cost / FLAGS.evaluation_epochs), flush=True)
                elif num_eval >= FLAGS.min_num_eval and not improved:
                    print ("no improve during curriculum {} --> stop".format(curriculum_idx))
                    break
                    
            #gt_final = sess.run(gt)
            #print('Final ground truth appliance data (length=', gt_final.size, ') has mean of ' + 
            #      str(np.mean(gt_final)) + ' and std of ' + str(np.std(gt_final)) + '.')
            #pred_final = sess.run(pred)
            #print('Final predicted appliance data (length=', pred_final.size, ') has mean of ' + 
            #      str(np.mean(pred_final)) + ' and std of ' + str(np.std(pred_final)) + '.')
                  
        if FLAGS.save:
            print('VAR_X: ', type(var_x))
            for i in range(len(var_x)):
                v = sess.run(var_x[i])
                print('SAVE VAR: ', str(v))
                np.save('./nilm_models/' + var_x[i].op.name.replace('/', '-'), v)
#             v.numpy().save('./models/nilm_model_weights.npy')
#         tf.train.Saver().save(sess, './modesl/nilm_model')

        run_time = timer() - start_time
        pipeline_util.log_pipeline_run(mode='train')
        print ("total time = {}s...".format(run_time))
        
        

if __name__ == "__main__":
    tf.app.run()
