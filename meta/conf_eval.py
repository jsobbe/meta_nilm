# Files
META_MODEL_PATH = './meta/models/_meta/'
NILM_MODEL_PATH = './meta/models/_nilm/'
OUTPUT_PATH = './meta/results/1_eval/'
SAVE_MODEL = True

# Experimental setup
NUM_EPOCHS = 2
NUM_STEPS = 500
LEARNING_RATE = 0.001

SEEDS = [
    41,# 22, 5, 31, #50
] 

# META setup
OPTIMIZERS = {
#       'rnn_base_e': {'path':META_MODEL_PATH + 'rnn_e_base_nbc/rp.l2l-0','shared_net':True} , #RNN_E
#       'rnn_base': {'path':META_MODEL_PATH + 'rnn_base_nb/rp.l2l-0','shared_net':True} ,
#       'rnn_no_scale': {'path':META_MODEL_PATH + 'rnn_no_scale/rp.l2l-0','shared_net':True} , #RNN_E
#       'rnn_double': {'path':[META_MODEL_PATH + 'rnn_e_base_nb_double/conv.l2l-0', META_MODEL_PATH + 'rnn_e_base_nb_double/fc.l2l-0'],'shared_net':False} , #RNN_E
#       'rnn_e_base_nb2': {'path':META_MODEL_PATH + 'rnn_e_base_nb2/rp.l2l-0','shared_net':True} ,
     'adam':{},
#     'sgd':{},
#     'momentum':{},
#     'adagrad':{},
#     'adadelta':{},
#     'rmsprop':{},
    
#      'dm_base': {'path':[META_MODEL_PATH + 'dm_base/conv.l2l-0', META_MODEL_PATH + 'dm_base/fc.l2l-0'],'shared_net':False} ,
#      'dm_e_base_nb': {'path':[META_MODEL_PATH + 'dm_e_base_nb/conv.l2l-0', META_MODEL_PATH + 'dm_e_base_nb/fc.l2l-0'],'shared_net':False} ,
#     'dm_base_single': {'path':META_MODEL_PATH + 'dm_base_single/cw.l2l-0','shared_net':True} ,
# #     'dm_e_base_fake': {'path':META_MODEL_PATH + 'dm_e_base/rp.l2l-0' ,'shared_net':True} ,
#     'dm_i_base': {'path':[META_MODEL_PATH + 'dm_i_base/conv.l2l-0', META_MODEL_PATH + 'dm_i_base/fc.l2l-0'],'shared_net':False} ,
#       'rnn_i_base_nb': {'path':META_MODEL_PATH + 'rnn_i_base_nb/rp.l2l-0','shared_net':True} ,
    
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
#     'fridge', 
    'kettle', 
    #'washing machine', 
    #'dish washer',
    #'oven', 
    #'microwave', 
]


# RNNPROP
BETA_1 = 0.95
BETA_2 = 0.95
