# Files
META_MODEL_PATH = './meta/models/_meta/'
NILM_MODEL_PATH = './meta/models/_nilm/'
OUTPUT_PATH = './meta/results/1_eval/'
SAVE_MODEL = True

# Experimental setup
NUM_RUNS = 1
NUM_EPOCHS = 7
NUM_STEPS = 100
LEARNING_RATE = 0.001

# META setup
OPTIMIZERS = {
    'adam':'',
    #'adagrad':'',
    #'momentum':'',
    #'rmsprop':'',
    #'dm': [META_MODEL_PATH + 'dm/conv.l2l-0', META_MODEL_PATH + 'dm/fc.l2l-0'], 
    #'dm': META_MODEL_PATH + 'dm/cw.l2l-0', 
    #'dme': [META_MODEL_PATH + 'dme/conv.l2l-0', META_MODEL_PATH + 'dme/fc.l2l-0'], 
    #'rnn': META_MODEL_PATH + 'rnnprop/rp.l2l-0'
}
PROBLEM = 'nilm_seq'
APPLIANCES = [
    'fridge', 
    #'washing machine', 
    #'dish washer',
    #'oven', 
    #'microwave'
]


# RNNPROP
BETA_1 = 0.95
BETA_2 = 0.95
