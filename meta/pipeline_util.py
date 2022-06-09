import conf_nilm
import conf_eval
import conf_train

import pandas as pd

PATH = './meta/logs/runs/'

# TODO datasets
def log_pipeline_run(mode, optimizer=None, runtime=None, final_cost=None, result=None):
    try:
        df = pd.read_pickle(PATH + 'log_' + mode + '.pkl')
    except FileNotFoundError:
        df = pd.DataFrame()
        
    if mode == 'train':
        log = get_meta_training_log()
    elif mode == 'eval':
        log = get_meta_evaluation_log()
    else:
        log = get_nilm_testing_log()
    
    if runtime:
        log['runtime'] = runtime
    if final_cost:
        log['final cost'] = final_cost
    if optimizer:
        log['optimizer'] = optimizer
    if result:
        log['result'] = result
        
    df = df.append(log, ignore_index=True)
    df.to_pickle(PATH + 'log_' + mode + '.pkl')
    df.to_csv(PATH + 'log_' + mode + '.csv')
        
    

def get_meta_training_log():
    experiment_meta = {'epochs':conf_train.NUM_EPOCHS,
                       'steps':conf_train.NUM_STEPS,
                       'appliances':','.join(conf_train.APPLIANCES), 
                        'unroll len':conf_train.UNROLL_LENGTH,
                       'learn_rate':conf_train.LEARNING_RATE,
                       'use imitation':conf_train.USE_IMITATION,
                       'use curriculum':conf_train.USE_CURRICULUM
                      }
    experiment_meta.update(get_nilm_model_log())
    return experiment_meta
    
    
def get_meta_evaluation_log():
    experiment_meta = {'epochs':conf_eval.NUM_EPOCHS,
                       'steps':conf_eval.NUM_STEPS,
                       'appliances':','.join(conf_eval.APPLIANCES), 
                        'number of runs':conf_eval.NUM_RUNS,
                       'learn_rate':conf_eval.LEARNING_RATE
                      }
    experiment_meta.update(get_nilm_model_log())
    return experiment_meta
    
    
def get_nilm_testing_log():
    experiment_meta = {'epochs':conf_nilm.NUM_EPOCHS,
                       'steps':conf_nilm.NUM_STEPS, 
                       'metrics':','.join(conf_nilm.METRICS), 
                        'unroll len':conf_nilm.UNROLL_LENGTH,
                       'learn_rate':conf_nilm.LEARNING_RATE
                      }
    experiment_meta.update(get_nilm_model_log())
    return experiment_meta
    
    
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
