# Files
META_MODEL_PATH = './meta/models/_meta/'
NILM_MODEL_PATH = './meta/models/_nilm/'
OUTPUT_PATH = './meta/results/1_eval/'
SAVE_MODEL = True

# Experimental setup
NUM_EPOCHS = 5
NUM_STEPS = 400
LEARNING_RATE = 0.001

SEEDS = [
    41, #22, 5, 31, #50
]

# META setup
OPTIMIZERS = {
#     'adam':{},
#     'sgd':{},
#     'momentum':{},
#     'adagrad':{},
#     'adadelta':{},
#     'rmsprop':{},
    
#     'dm_e_base': {'path':[META_MODEL_PATH + 'dm_e_base/conv.l2l-0', META_MODEL_PATH + 'dm_e_base/fc.l2l-0'],'shared_net':False} ,
#     'dm_base_single': {'path':META_MODEL_PATH + 'dm_base_single/cw.l2l-0','shared_net':True} ,
# #     'dm_e_base_fake': {'path':META_MODEL_PATH + 'dm_e_base/rp.l2l-0' ,'shared_net':True} ,
      'rnn_base': {'path':META_MODEL_PATH + 'rnn_base/rp.l2l-0','shared_net':True} ,
      'rnn_no_batch': {'path':META_MODEL_PATH + 'rnn_base/rp.l2l-0','shared_net':True} ,
#     'dm_base': {'path':[META_MODEL_PATH + 'dm_base/conv.l2l-0', META_MODEL_PATH + 'dm_base/fc.l2l-0'],'shared_net':False} ,
#     'dm_i_base': {'path':[META_MODEL_PATH + 'dm_i_base/conv.l2l-0', META_MODEL_PATH + 'dm_i_base/fc.l2l-0'],'shared_net':False} ,
#     'rnn_e_base': {'path':META_MODEL_PATH + 'rnn_e_base/rp.l2l-0','shared_net':True} ,
    'rnn_i_base': {'path':META_MODEL_PATH + 'rnn_i_base/rp.l2l-0','shared_net':True} ,
    
#     'l2o_dm': {'path':[META_MODEL_PATH + 'dm/conv.l2l-0', META_MODEL_PATH + 'dm/fc.l2l-0'],'shared_net':False} ,
#     'l2o_dm_e': {'path':[META_MODEL_PATH + 'conv.l2l-0', META_MODEL_PATH + 'fc.l2l-0'],'shared_net':False} ,
    #'l2o_dm': {'path':META_MODEL_PATH + 'dm/cw.l2l-0'}, 
    #'l2o_dme': {'path':[META_MODEL_PATH + 'dme/conv.l2l-0', META_MODEL_PATH + 'dme/fc.l2l-0']}, 
    #'l2o_rnn_days': {'path':META_MODEL_PATH + 'rp.l2l-0'},
    #'l2o_rnn': {'path':META_MODEL_PATH + 'rnnprop/rp.l2l-0','shared_net':True},
    #'l2o_rnn_e': {'path':META_MODEL_PATH + 'rnnprop_e/rp.l2l-0','shared_net':True},
    #'l2o_rnn_appls': {'path':META_MODEL_PATH + 'rnn/rp.l2l-0'}
}
PROBLEM = 'nilm_seq'
APPLIANCES = [
#     'stove', 
#     'lighting', 
#     'washer dryer',
#     'kitchen',
    'fridge', 
    #'washing machine', 
    #'dish washer',
    #'oven', 
    #'microwave', 
]


# RNNPROP
BETA_1 = 0.95
BETA_2 = 0.95
