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

from six.moves import xrange
from tensorflow.contrib.learn.python.learn import monitored_session as ms
import tensorflow as tf

import meta_dm_eval as meta
import nilm_config
import util
import numpy as np
import matplotlib.pyplot as plt

from eval_nilm import nilm_eval

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("optimizer", "dm", "Optimizer.")
flags.DEFINE_string("problem", "simple", "Type of problem.")

flags.DEFINE_string("path", None, "Path to saved meta-optimizer network.")
flags.DEFINE_string("output_path", None, "Path to output results.")
flags.DEFINE_boolean("save", False, "Whether to save the resulting nilm-model.")

flags.DEFINE_integer("num_epochs", 1, "Number of evaluation epochs.")
flags.DEFINE_integer("num_steps", 10000, "Number of optimization steps per epoch.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_integer("seed", None, "Seed for TensorFlow's RNG.")


def main(_):
    np.set_printoptions(precision=3)
    # Configuration.
    num_unrolls = FLAGS.num_steps
    if FLAGS.seed:
        tf.set_random_seed(FLAGS.seed)

    # Problem, NET_CONFIG = predefined conf for META-net, NET_ASSIGNMENTS = None
    problem, net_config, net_assignments = util.get_config(FLAGS.problem, ['./meta/models/conv.l2l-0', './meta/models/fc.l2l-0'])
    
    print('NET_CONFIG: ', net_config)

    # Optimizer setup.
    if FLAGS.optimizer == "adam":
        cost_op, gt, pred = problem()
        problem_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        problem_reset = tf.variables_initializer(problem_vars)

        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        optimizer_reset = tf.variables_initializer(optimizer.get_slot_names())
        update = optimizer.minimize(cost_op)
        reset = [problem_reset, optimizer_reset]
    elif FLAGS.optimizer == "dm":
        #if FLAGS.path is None:
        #    logging.warning("Evaluating untrained L2L optimizer")
        optimizer = meta.MetaOptimizer(**net_config)
        meta_loss, fx_array, problem_vars, s_final, gt, pred = optimizer.meta_loss(problem, 1, net_assignments=net_assignments)
        _, update, reset, cost_op, _ = meta_loss
    else:
        raise ValueError("{} is not a valid optimizer".format(FLAGS.optimizer))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with ms.MonitoredSession() as sess:
        sess.run(reset)
        # Prevent accidental changes to the graph.
        tf.get_default_graph().finalize()

        total_time = 0
        total_cost = 0
        loss_record = []
        for e in xrange(FLAGS.num_epochs):
            # Training.
            time, cost, result = util.run_eval_epoch(sess, cost_op, [update], num_unrolls)
            total_time += time
            total_cost += sum(cost) / num_unrolls
            loss_record += cost
        
        
        print('avg_cost:', total_cost)
        print('loss_record:', loss_record)
        print('final_loss:', cost[-1])
        # Results.
        util.print_stats("Epoch {}".format(FLAGS.num_epochs), total_cost,
                         total_time, FLAGS.num_epochs)
        
        if FLAGS.output_path is not None:
            if not os.path.exists(FLAGS.output_path):
                os.mkdir(FLAGS.output_path)
        output_file = '{}/{}_eval_loss_record.pickle-{}'.format(FLAGS.output_path, FLAGS.optimizer, FLAGS.problem)
        with open(output_file, 'wb') as l_record:
            pickle.dump(loss_record, l_record)
        print("Saving evaluate loss record {}".format(output_file))
        
        plt.figure()
        plt.plot(loss_record,label=FLAGS.optimizer)
        plt.legend()
        plt.xlabel('Steps')
        plt.ylabel('Training Loss')
        plt.yscale("log")
        plt.savefig('./meta/results/' + FLAGS.optimizer + '.png')

        
        gt_final = sess.run(gt)
        print('Final ground truth appliance data has mean of ' + 
              str(np.mean(gt_final)) + ' and std of ' + str(np.std(gt_final)) + '.')
        pred_final = sess.run(pred)
        print('Final predicted appliance data has mean of ' + 
              str(np.mean(pred_final)) + ' and std of ' + str(np.std(pred_final)) + '.')
        
        result_df = pd.DataFrame({'gt': gt_final, 'pred': pred_final})
        result_df.head(20)
        result_df.to_csv('./meta/results/adam_test.csv')
        
        if FLAGS.save:
            print('VAR_X: ', type(problem_vars))
            print('VAR_X length: ', len(problem_vars))
            print('VAR_X content: ', str(problem_vars))
            count = 0
            for i in range(len(problem_vars)):
                print(problem_vars[i].name, ': ', count, ' => ', nilm_config.NILM_VARS[count])
                if not 'batch_norm' in problem_vars[i].name:
                    v = sess.run(problem_vars[i])
                    np.save('./nilm_models/eval/' + FLAGS.optimizer + '/' + nilm_config.NILM_VARS[count], v)
                    count += 1
    


if __name__ == "__main__":
    tf.app.run()
