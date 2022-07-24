# Files
META_MODEL_PATH = './meta/models/_meta/'
NILM_MODEL_PATH = './meta/models/_nilm/'
OUTPUT_PATH = './meta/results/1_eval/'
SAVE_MODEL = True

# Experimental setup
NUM_EPOCHS = 4
NUM_STEPS = 500
# NUM_RUNS = 9
SEEDS = [ 
    6, 15, 37, 82, 
#     50, 93, 22, 14, 
#     71
] # number of seeds implies number of evaluation runs

# META setup
OPTIMIZERS = {
    # GD variants
    'adam':{'path_postfix':'opt'},
    'sgd':{'path_postfix':'opt'},
    'momentum':{'path_postfix':'opt'},
    'adagrad':{'path_postfix':'opt'},
    'adadelta':{'path_postfix':'opt'},
    'rmsprop':{'path_postfix':'opt'},
    
    # BATCH
#     'adam':{'path_postfix':'batch'},
#       'rnn_base_b': {'path':META_MODEL_PATH + 'rnn_base_b/rp.l2l-0','shared_net':True, 'path_postfix':'batch'} ,
#       'rnn_e_base_b': {'path':META_MODEL_PATH + 'rnn_e_base_b/rp.l2l-0','shared_net':True, 'path_postfix':'batch'} ,
#      'dm_i_base_b': {'path':[META_MODEL_PATH + 'dm_i_base_b/conv.l2l-0', META_MODEL_PATH + 'dm_i_base_b/fc.l2l-0'],'shared_net':False, 'path_postfix':'batch'} ,
#      'dm_base_b': {'path':[META_MODEL_PATH + 'dm_base_b/conv.l2l-0', META_MODEL_PATH + 'dm_base_b/fc.l2l-0'],'shared_net':False, 'path_postfix':'batch'} ,
#      'dm_e_base_b': {'path':[META_MODEL_PATH + 'dm_e_base_b/conv.l2l-0', META_MODEL_PATH + 'dm_e_base_b/fc.l2l-0'],'shared_net':False, 'path_postfix':'batch'} ,
    
    #NO BATCH
    # base
#       'rnn_e_base': {'path':META_MODEL_PATH + 'rnn_e_base_nb2/rp.l2l-0','shared_net':True, 'path_postfix':'base'} ,
#       'rnn_base': {'path':META_MODEL_PATH + 'rnn_base_nb/rp.l2l-0','shared_net':True, 'path_postfix':'base'} ,
#     'adam':{'path_postfix':'base'},
    # appls
#       'rnn_e_appls': {'path':META_MODEL_PATH + 'rnn_e_appls/rp.l2l-0','shared_net':True, 'path_postfix':'appls'} ,
#       'rnn_appls': {'path':META_MODEL_PATH + 'rnn_appl/rp.l2l-0','shared_net':True, 'path_postfix':'appls'} ,
#     'adam':{'path_postfix':'appls'},
    # data
#       'rnn_e_data': {'path':META_MODEL_PATH + 'rnn_e_data/rp.l2l-0','shared_net':True, 'path_postfix':'data'} ,
#       'rnn_data': {'path':META_MODEL_PATH + 'rnn_data/rp.l2l-0','shared_net':True, 'path_postfix':'data'} ,
#     'adam':{'path_postfix':'data'},
    # var all
#       'rnn_e_both': {'path':META_MODEL_PATH + 'rnn_e_both/rp.l2l-0','shared_net':True, 'path_postfix':'both'} ,
#       'rnn_both': {'path':META_MODEL_PATH + 'rnn_both/rp.l2l-0','shared_net':True, 'path_postfix':'both'} ,
#     'adam':{'path_postfix':'both'},
    
    #NO BATCH iAWE
    # base
#       'rnn_e_base': {'path':META_MODEL_PATH + 'rnn_e_base_nb2/rp.l2l-0','shared_net':True, 'path_postfix':'base_iAWE_b'} ,
#       'rnn_base': {'path':META_MODEL_PATH + 'rnn_base_nb/rp.l2l-0','shared_net':True, 'path_postfix':'base_iAWE_b'} ,
#     'adam':{'path_postfix':'base_iAWE_b'},
    # appls
#       'rnn_e_appls': {'path':META_MODEL_PATH + 'rnn_e_appls/rp.l2l-0','shared_net':True, 'path_postfix':'appls_iAWE_b'} ,
#       'rnn_appls': {'path':META_MODEL_PATH + 'rnn_appl/rp.l2l-0','shared_net':True, 'path_postfix':'appls_iAWE_b'} ,
#     'adam':{'path_postfix':'appls_iAWE_b'},
    # data
#       'rnn_e_data': {'path':META_MODEL_PATH + 'rnn_e_data/rp.l2l-0','shared_net':True, 'path_postfix':'data_iAWE_b'} ,
#       'rnn_data': {'path':META_MODEL_PATH + 'rnn_data/rp.l2l-0','shared_net':True, 'path_postfix':'data_iAWE_b'} ,
#     'adam':{'path_postfix':'data_iAWE_b'},
    # var all 
#       'rnn_e_both': {'path':META_MODEL_PATH + 'rnn_e_both/rp.l2l-0','shared_net':True, 'path_postfix':'both_iAWE_b'} ,
#       'rnn_both': {'path':META_MODEL_PATH + 'rnn_both/rp.l2l-0','shared_net':True, 'path_postfix':'both_iAWE_b'} ,
#     'adam':{'path_postfix':'both_iAWE_b'},
    
}
PROBLEM = 'nilm_seq' #TODO remove
APPLIANCES = [
#     'stove', 
#     'lighting', 
#     'washer dryer',
#     'kitchen',
    'fridge', 
#     'kettle', 
#     'air conditioner',
#     'water kettle', 
    #'washing machine', 
#     'dish washer',
#     'oven', 
    #'microwave', 
]


# Analytic optimizers
LEARNING_RATE = 0.001 
BETA_1 = 0.95
BETA_2 = 0.95
