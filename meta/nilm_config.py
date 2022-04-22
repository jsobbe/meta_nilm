DATASETS_TRAIN = {'redd':{
    'path': './data/redd.h5',
    'buildings': {
        2: {'start_time': '2011-04-01', 'end_time': '2011-05-28'}
    }}}
DATASETS_EVAL = {'redd':{
    'path': './data/redd.h5',
    'buildings': {
        1: {'start_time': '2011-04-01', 'end_time': '2011-04-30'}
    }}}
DATASETS_TEST = {'redd':{
    'path': './data/redd.h5',
    'buildings': {
        1: {'start_time': '2011-05-01', 'end_time': '2011-05-02'}
    }}}
POWER = {'mains': ['apparent'], 'appliance': ['active']}
APPLIANCES = ['fridge']
APPLIANCE_PARAMS = {'fridge': {'mean':200, 'std':400}}
DROP_NANS = True
WINDOW_SIZE = 599 # According to seq paper
SAMPLE_PERIOD = 3
BATCH_SIZE=512
ARTIFICIAL_AGGREGATE = False # TODO Check what it does and what is better?
BATCH_NORM = False # TODO from meta. Does it make sense on top of normalization done by NILMTK?

# For NILM evaluation
METRICS = ['mae']
PREPROCESSING = True
DISPLAY_PRED = True
OPTIMIZERS = ['L2L', 'Adam']