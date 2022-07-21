import conf_nilm
import conf_eval
import conf_train
import conf_data

import pandas as pd
import datetime

PATH = './meta/logs/runs/'

# TODO datasets
def log_pipeline_run(mode, optimizer=None, runtime=None, final_loss=None, avg_loss=None, result=None, metrics=None):
    try:
        df = pd.read_pickle(PATH + 'log_' + mode + '_final.pkl')
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
    if final_loss:
        log['final loss'] = final_loss
    if avg_loss:
        log['avg loss'] = avg_loss
    if optimizer:
        log['optimizer'] = optimizer
    if result:
        log['result'] = result
    if metrics:
        log['metrics'] = metrics
    log['date'] = datetime.datetime.now().strftime("%c")
        
    df = df.append(log, ignore_index=True)
    df.to_pickle(PATH + 'log_' + mode + '_final.pkl')
    df.to_csv(PATH + 'log_' + mode + '_final.csv', sep=';')
        
    

def get_meta_training_log():
    experiment_meta = {'epochs':conf_train.NUM_EPOCHS,
                       'steps':conf_train.NUM_STEPS,
                       'appliances':','.join(conf_train.APPLIANCES), 
                        'unroll len':conf_train.UNROLL_LENGTH,
                       'learn_rate':conf_train.LEARNING_RATE,
                       'use imitation':conf_train.USE_IMITATION,
                       'use curriculum':conf_train.USE_CURRICULUM,
                       'shared net':conf_train.SHARED_NET,
                       'random scaling':conf_train.USE_SCALE,
                       'data':conf_data.DATASETS_TRAIN,
                       'model':conf_train.SAVE_PATH + conf_train.OPTIMIZER_NAME
                      }
    experiment_meta.update(get_nilm_model_log())
    return experiment_meta
    
    
def get_meta_evaluation_log():
    experiment_meta = {'epochs':conf_eval.NUM_EPOCHS,
                       'steps':conf_eval.NUM_STEPS,
                       'appliances':','.join(conf_eval.APPLIANCES), 
                       'learn_rate':conf_eval.LEARNING_RATE,
                       'data':conf_data.DATASETS_EVAL
                      }
    experiment_meta.update(get_nilm_model_log())
    return experiment_meta
    
    
def get_nilm_testing_log():
    experiment_meta = {'metrics':','.join(conf_nilm.METRICS), 
                       'data':conf_data.DATASETS_TEST
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
                       'preprocessing':conf_nilm.PREPROCESSING
    }
