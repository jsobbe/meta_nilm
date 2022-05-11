# Experimental setup
NUM_RUNS = 3
NUM_EPOCHS = 1
NUM_STEPS = 250
LEARNING_RATE = 0.001

# META setup
OPTIMIZERS = {
    'adam':'',
    'adagrad':'',
    'momentum':'',
    'rmsprop':'',
    #'dm': ['./meta/models/dm/conv.l2l-0', './meta/models/dm/fc.l2l-0'], 
    #'dm': './meta/models/dm/cw.l2l-0', 
    #'dme': ['./meta/models/dme/conv.l2l-0', './meta/models/dme/fc.l2l-0'], 
    #'rnn': './meta/models/rnn/rp.l2l-0'
}
PROBLEM = 'nilm_seq'

# Files
OUTPUT_PATH = './meta/results/'
SAVE_MODEL = False

# RNNPROP
BETA_1 = 0.95
BETA_2 = 0.95
