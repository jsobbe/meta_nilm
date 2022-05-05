DATASETS_TRAIN = {'redd':{
    'path': './data/redd.h5',
    'buildings': {
        2: {'start_time': '2011-04-01', 'end_time': '2011-05-10'}
    }}}
DATASETS_EVAL = {'redd':{
    'path': './data/redd.h5',
    'buildings': {
        1: {'start_time': '2011-04-01', 'end_time': '2011-05-10'}
    }}}
DATASETS_TEST = {'redd':{
    'path': './data/redd.h5',
    'buildings': {
        1: {'start_time': '2011-05-11', 'end_time': '2011-05-12'}
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

# For NILM evaluation
METRICS = ['mae']
DISPLAY_PRED = True
OPTIMIZERS = ['L2L', 'Adam']