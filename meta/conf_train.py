SAVE_PATH = "./meta/models/_meta/"
OUTPUT_PATH = "./meta/results/0_train/"

# General training/model choices
PROBLEM = "nilm_seq"

# TODO add optimizer_name -> generate paths and create folders
OPTIMIZER_NAME = "dm_base_nb"

APPLIANCES = [
    'fridge', 
    #'washing machine', 
    #'oven', 
    #'microwave', 
    #'dish washer'
]

# Enhanced training
USE_IMITATION = False
USE_CURRICULUM = False

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

CONTINUE_TRAINING = False

USE_SCALE = True
RD_SCALE_BOUND = 3.0

# RnnProp parameters
BETA1 = 0.95
BETA2 = 0.95

# DM parameters
SHARED_NET = False


# Imitation technique
NUM_MT = 1
MT_OPTIMIZERS = 'adam'
MT_RATIO = 0.3
MT_RATIOS = "0.3 0.3 0.3"
K = 1