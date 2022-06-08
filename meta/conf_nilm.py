DATASETS_TRAIN = {'redd':{
    'path': './data/redd.h5',
    'buildings': {
        2: {'start_time': '2011-04-01', 'end_time': '2011-05-28'}
    }}}
DATASETS_EVAL = {'redd':{
    'path': './data/redd.h5',
    'buildings': {
        1: {'start_time': '2011-04-01', 'end_time': '2011-05-10'}
    }}}
DATASETS_TEST = {'redd':{
    'path': './data/redd.h5',
    'buildings': {
        1: {'start_time': '2011-05-11', 'end_time': '2011-05-30'}
    }}}
POWER = {'mains': ['apparent'], 'appliance': ['active']}
APPLIANCES = ['fridge']
DROP_NANS = True
WINDOW_SIZE = 599 # According to seq paper
SAMPLE_PERIOD = 3
BATCH_SIZE=128
ARTIFICIAL_AGGREGATE = False # TODO Check what it does and what is better?
BATCH_NORM = True # TODO from meta. Does it make sense on top of normalization done by NILMTK?
PREPROCESSING = True
SHARED_NET = True

# For NILM evaluation
METRICS = ['mae', 
          'rmse',
          'f1score',
          'nep']
DISPLAY_PRED = True
OPTIMIZERS = [
    #'dm', 
    'dme', 
    'rnn', 
    'adam', 
    'rmsprop']

NILM_VARS = ['conv_1-weights','conv_1-biases',
              'conv_2-weights','conv_2-biases',
              'conv_3-weights','conv_3-biases',
              'conv_4-weights','conv_4-biases',
              'conv_5-weights','conv_5-biases',
              'dense_1-weights','dense_1-biases',
              'dense_2-weights','dense_2-biases']

OUTPUT_PATH = './meta/results/2_nilm/'
MODEL_PATH = './meta/models/_nilm/'