SAVE_PATH = "./meta/models/_meta/"
OUTPUT_PATH = "./meta/results/0_train/"

# General training/model choices
PROBLEM = "nilm_seq"
OPTIMIZERS = [
    #'dm', 
    #'dm_e', 
    'rnnprop',
    #'rnnprop_e'
]
APPLIANCES = [
    'fridge', 
    #'washing machine', 
    #'oven', 
    #'microwave'
]

# Enhanced training
USE_IMITATION = False
USE_CURRICULUM = False

# Training parameters
NUM_EPOCHS = 400
VALIDATION_PERIOD = 50
VALIDATION_EPOCHS = 5
NUM_STEPS = 150
UNROLL_LENGTH = 20
LEARNING_RATE = 0.001
SECOND_DERIVATIVES = False

CONTINUE_TRAINING = False

USE_SCALE = False
RD_SCALE_BOUND = 3.0

# RnnProp parameters
BETA1 = 0.95
BETA2 = 0.95

# DM parameters

# Curriculum parameters
MIN_NUM_EVAL = 3

# Imitation technique
NUM_MT = 1
MT_OPTIMIZERS = 'adam'
MT_RATIO = 0.3
MT_RATIOS = "0.3 0.3 0.3"
K = 1