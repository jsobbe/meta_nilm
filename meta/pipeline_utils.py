import conf_nilm
import conf_eval
import conf_nilm

import pandas as pd

PATH = './meta/logs/'

# TODO datasets
def log_pipeline_run(mode):
    df = pd.read_pickle(PATH + 'log_' + mode + '.pkl')
    if not df:
        df = pd.DataFrame()
        
    if mode == 'train':
        log = get_meta_training_log()
    elif mode == 'eval':
        log = get_meta_evaluation_log()
    else:
        log = get_nilm_testing_log()
    df.append(log)
        
    

def get_meta_training_log():
    experiment_meta = {'epochs':conf_train.NUM_EPOCHS,
                       'steps':conf_train.NUM_STEPS,
                       'optimizers':','.join(conf_train.OPTIMIZERS), 
                       'appliances':','.join(conf_train.APPLIANCES), 
                        'unroll len':conf_train.UNROLL_LENGTH,
                       'learn_rate':conf_train.LEARNING_RATE,
                       'use imitation':conf_train.USE_IMITATION,
                       'use curriculum':conf_train.USE_CURRICULUM
                      }
    experiment_meta.append(get_nilm_model_log())
    
    
def get_meta_evaluation_log():
    experiment_meta = {'epochs':conf_eval.NUM_EPOCHS,
                       'steps':conf_eval.NUM_STEPS,
                       'optimizers':','.join(conf_eval.OPTIMIZERS), 
                       'appliances':','.join(conf_eval.APPLIANCES), 
                        'unroll len':conf_eval.UNROLL_LENGTH,
                       'learn_rate':conf_eval.LEARNING_RATE
                      }
    experiment_meta.append(get_nilm_model_log())
    
    
def get_nilm_testing_log():
    experiment_meta = {'epochs':conf_nilm.NUM_EPOCHS,
                       'steps':conf_nilm.NUM_STEPS,
                       'optimizers':','.join(conf_nilm.OPTIMIZERS), 
                       'metrics':','.join(conf_nilm.METRICS), 
                        'unroll len':conf_nilm.UNROLL_LENGTH,
                       'learn_rate':conf_nilm.LEARNING_RATE
                      }
    
    
def get_nilm_model_log():
    return {'power':conf_nilm.POWER,
                       'drop nans':conf_nilm.DROP_NANS,
                       'window size':conf_nilm.WINDOW_SIZE,
                       'sample period':conf_nilm.SAMPLE_PERIOD,
                       'batch size':conf_nilm.BATCH_SIZE,
                       'batch norm':conf_nilm.BATCH_NORM,
                       'art aggregate':conf_nilm.ARTIFICIAL_AGGREGATE,
                       'shared net':conf_nilm.SHARED_NET,
                       'preprocessing':conf_nilm.PREPROCESSING,
    }
