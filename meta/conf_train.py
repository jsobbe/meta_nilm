SAVE_PATH = "./meta/models/_meta/"
OUTPUT_PATH = "./meta/results/0_train/"

# General training/model choices
PROBLEM = "nilm_seq"

OPTIMIZER_NAME = "rnn_e_appls"

APPLIANCES = [
    'fridge', 
    'oven', 
    'dish washer'
]

# Enhanced training
USE_IMITATION = True
USE_CURRICULUM = True

# Training parameters
NUM_EPOCHS = 800
VALIDATION_PERIOD = 50
VALIDATION_EPOCHS = 9
NUM_STEPS = 400
UNROLL_LENGTH = 20
LEARNING_RATE = 0.001
SECOND_DERIVATIVES = False
# Curriculum parameters
MIN_NUM_EVAL = 3

CONTINUE_TRAINING = True

USE_SCALE = True
RD_SCALE_BOUND = 3.0

# RnnProp parameters
BETA1 = 0.95
BETA2 = 0.95

# DM parameters
SHARED_NET = True


# Imitation technique
NUM_MT = 1
MT_OPTIMIZERS = 'adam'
MT_RATIO = 0.3
MT_RATIOS = "0.3 0.3 0.3"
K = 1