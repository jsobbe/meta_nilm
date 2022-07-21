SAVE_PATH = "./meta/models/_meta/"
OUTPUT_PATH = "./meta/results/0_train/"

# General training/model choices
PROBLEM = "nilm_seq"

OPTIMIZER_NAME = "test"

APPLIANCES = [
    'fridge', 
    'oven', 
    'dish washer', 
    'kettle'
]

# Training parameters
NUM_EPOCHS = 800
VALIDATION_PERIOD = 50
VALIDATION_EPOCHS = 5
NUM_STEPS = 400
UNROLL_LENGTH = 20
LEARNING_RATE = 0.001
SECOND_DERIVATIVES = False

# Enhanced training
USE_IMITATION = False
USE_CURRICULUM = False

# Curriculum parameters
MIN_NUM_EVAL = 3

CONTINUE_TRAINING = False

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