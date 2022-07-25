# ----------------- NILM Evaluation -----------------
# Files
OUTPUT_PATH = './meta/results/2_nilm/'
MODEL_PATH = './meta/models/_nilm/'

# Evaluation parameters
APPLIANCES = ['fridge']
METRICS = ['mae', 
          'rmse',
          'nep']
OPTIMIZERS = [
#     'sgd',
#     'momentum',
#     'adagrad',
#     'adadelta',
#     'rmsprop',
#     'adam',
#     'rnn_both/both',
#     'rnn_e_both/both',
#     'adam/both',
    'rnn_base/base',
    'rnn_e_base_nb2/base',
    'adam/base',
#     'dish washer/adam', 
#     'dish washer/rnn_base', 
#     'dish washer/rnn_e_base_nb2', 
]
DISPLAY_PRED = True
DISPLAY_DETAIL_TIME = {'start_time': '2011-04-21 06:00:00', 'end_time': '2011-04-21 12:00:00'}



# ----------------- S2P Network & Preprocessing -----------------
POWER = {'mains': ['apparent'], 'appliance': ['active']}
DROP_NANS = True
WINDOW_SIZE = 599
SAMPLE_PERIOD = 3
BATCH_SIZE=512
ARTIFICIAL_AGGREGATE = False
BATCH_NORM = False
PREPROCESSING = True

# Network util for saving nilm models
NILM_VARS_BATCH_NORM = ['conv_1-weights','conv_1-biases',
             'conv_1-gamma', 'conv_1-beta',
              'conv_2-weights','conv_2-biases',
             'conv_2-gamma', 'conv_2-beta',
              'conv_3-weights','conv_3-biases',
             'conv_3-gamma', 'conv_3-beta',
              'conv_4-weights','conv_4-biases',
             'conv_4-gamma', 'conv_4-beta',
              'conv_5-weights','conv_5-biases',
             'conv_5-gamma', 'conv_5-beta',
              'dense_1-weights','dense_1-biases',
              'dense_2-weights','dense_2-biases']
NILM_VARS = ['conv_1-weights','conv_1-biases',
              'conv_2-weights','conv_2-biases',
              'conv_3-weights','conv_3-biases',
              'conv_4-weights','conv_4-biases',
              'conv_5-weights','conv_5-biases',
              'dense_1-weights','dense_1-biases',
              'dense_2-weights','dense_2-biases']
