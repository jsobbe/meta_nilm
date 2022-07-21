import conf_nilm
import conf_data
import nilm_seq2point
import tensorflow as tf
import os
import pickle

from nilmtk.dataset import DataSet
from nilmtk.losses import *
from nilmtk.metergroup import MeterGroup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.axes as ax
import matplotlib
import datetime
import pipeline_util

def _dropna(mains_df, appliance_dfs=[], drop=True):
    """
    Drops the missing values in the Mains reading and appliance readings and returns consistent data by computing the intersection
    """
    # The below steps are for making sure that data is consistent by doing intersection across appliances
    mains_df = mains_df.dropna()
    ix = mains_df.index
    mains_df = mains_df.loc[ix]
    for i in range(len(appliance_dfs)):
        if drop:
            appliance_dfs[i] = appliance_dfs[i].dropna()
        else:
            appliance_dfs[i] = appliance_dfs[i].fillna(0)

        
    for app_df in appliance_dfs:
        ix = ix.intersection(app_df.index)
    mains_df = mains_df.loc[ix]
    new_appliances_list = []
    for app_df in appliance_dfs:
        new_appliances_list.append(app_df.loc[ix])
    return mains_df,new_appliances_list


def _get_appliance_params(train_appliances):
    appliance_params = {}
    # Find the parameters using the first
    for (app_name,df_list) in train_appliances:
        l = np.array(pd.concat(df_list,axis=0))
        app_mean = np.mean(l)
        app_std = np.std(l)
        if app_std<1:
            app_std = 100
        appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})
    return appliance_params

def _make_plots(data, name):
    appliance = name.split('_')[-1]
    if not os.path.exists(conf_nilm.OUTPUT_PATH + appliance + '/'):
        os.mkdir(conf_nilm.OUTPUT_PATH + appliance + '/')
    
    font = {'family' : 'normal',
        'size'   : 16}
    matplotlib.rc('font', **font)
    myFmt = mdates.DateFormatter('%d.')
    plt.gca().xaxis.set_major_formatter(myFmt)
    
    plt.figure(figsize=(100, 12))
    for label, values in data.items():
        plt.plot(values, label=label)
    plt.title(name)
    plt.legend(fontsize=40)
    plt.ylim(0, 2000)
    plt.xlabel('Time', fontsize=40)
    plt.ylabel('Power (W)', fontsize=40)
    plt.savefig(conf_nilm.OUTPUT_PATH + appliance + '/' + name.replace('/', '') + '.png')

    plt.figure(figsize=(12, 6))
    for label, values in data.items():
        plt.plot(values[conf_nilm.DISPLAY_DETAIL_TIME['start_time'] : conf_nilm.DISPLAY_DETAIL_TIME['end_time']], label=label)
    plt.title(name)
    plt.legend()
    plt.ylim(0, 2000)
    plt.xlabel('Time')
    plt.ylabel('Power (W)')
    plt.savefig(conf_nilm.OUTPUT_PATH + appliance + '/' + name.replace('/', '') + '_detailed.png')
    
#     output_file = '{}{}/{}_nilm_results.pickle'.format(conf_nilm.OUTPUT_PATH, appliance, name)
#     with open(output_file, 'wb') as l_record:
#         pickle.dump(data, l_record)

class nilm_eval():
    
    def __init__(self):
        np.set_printoptions(precision=3)
    
        self.metrics = conf_nilm.METRICS
        self.appliances = conf_nilm.APPLIANCES
        self.drop_nans = conf_nilm.DROP_NANS
        self.power = conf_nilm.POWER
        self.appliances = conf_nilm.APPLIANCES
        self.window_size = conf_nilm.WINDOW_SIZE
        self.sample_period = conf_nilm.SAMPLE_PERIOD
        self.batch_size = conf_nilm.BATCH_SIZE
        self.artificial_aggregate = conf_nilm.ARTIFICIAL_AGGREGATE
        self.test_submeters = []
        self.errors = []
        self.errors_keys = []
        self.do_preprocessing = conf_nilm.PREPROCESSING
        self.display_predictions = conf_nilm.DISPLAY_PRED
        self.optimizers = conf_nilm.OPTIMIZERS

    def test(self):
        # store the test_main readings for all buildings
        d = conf_data.DATASETS_TEST

        for dataset in d:
            print("Loading data for ",dataset, " dataset")
            test=DataSet(d[dataset]['path'])
            for building in d[dataset]['buildings']:
                test.set_window(start=d[dataset]['buildings'][building]['start_time'],
                                end=d[dataset]['buildings'][building]['end_time'])
                test_mains=next(test.buildings[building].elec.mains().load(physical_quantity='power', 
                                                                           ac_type=self.power['mains'],
                                                                           sample_period=self.sample_period))
                appliance_readings=[]

                for appliance in self.appliances:
                    test_df=next((test.buildings[building].elec[appliance].load(
                        physical_quantity='power', ac_type=self.power['appliance'], 
                        sample_period=self.sample_period)))
                    appliance_readings.append(test_df)

#                 if self.drop_nans:
                test_mains, appliance_readings = _dropna(test_mains,appliance_readings, drop=self.drop_nans)
                print('appliance_readings: ', appliance_readings)
                    
                if not appliance_readings:
                    raise ValueError('No appliance data found for specified appliances and time.')

                if self.artificial_aggregate:
                    print ("Creating an Artificial Aggregate")
                    test_mains = pd.DataFrame(np.zeros(appliance_readings[0].shape),
                                              index = appliance_readings[0].index,
                                              columns=appliance_readings[0].columns)
                    for app_reading in appliance_readings:
                        test_mains+=app_reading
                        
                for i, appliance_name in enumerate(appliance_readings):
                    self.test_submeters.append((self.appliances[i],[appliance_readings[i]]))
                self.appliance_params = _get_appliance_params(self.test_submeters)

                self.test_mains = [test_mains]
                self.storing_key = str(dataset) + "_" + str(building) 
                self.call_predict(test.metadata["timezone"])



    def predict(self, test_elec, test_submeters, sample_period, timezone ):
        """
        Generates predictions on the test dataset using the specified classifier.
        """
        if self.display_predictions:
            mains = None
            for main in test_elec:
                if not mains:
                    mains = main
                else:
                    mains.append(main, ignore_index=True)
        _make_plots(data={'Mains':mains}, name='mains')
                    
        # "ac_type" varies according to the dataset used. 
        # Make sure to use the correct ac_type before using the default parameters in this code.   
        if self.do_preprocessing:
            test_main_list, _ = nilm_seq2point.preprocess_data(test_elec, submeters_lst=None, window_size=self.window_size)
            
        mains = []
        mains_len = 0
        for main_df in test_main_list:
            if not main_df.empty:
                mains_len += len(main_df)
            mains.append(main_df.to_numpy())
        if mains_len <= 1:
            raise ValueError('No mains data found in provided time frame') 
        print('num of mains:', mains_len)
        mains_np = np.asarray(mains)

        test_predictions = []
        disggregation_dict = {}
        for optimizer in self.optimizers:
            for appliance in self.appliances:
                print("=========== PREDICTION for | ", appliance, " | ", optimizer, " | =============")
                with tf.Session() as sess:
                    model_path = conf_nilm.MODEL_PATH + appliance + '/' + optimizer + '/'
                    
                    problem, mains_p, _ = nilm_seq2point.model(mode='nilm_test', model_path=model_path, optimizer=optimizer, batch_size=mains_len, predict=True)
                    result = problem()
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())
                    
                    prediction = sess.run(result, feed_dict={mains_p:mains_np.squeeze()})
                    print('Add appl mean: ', self.appliance_params[appliance]['mean'])
                    print('Add appl std: ', self.appliance_params[appliance]['std'])
                    print('Before adjusting prediction mean: ', np.mean(prediction))
                    print('Before adjusting prediction std: ', np.std(prediction))
                    prediction = self.appliance_params[appliance]['mean'] + prediction * self.appliance_params[appliance]['std']
                    print('After adjusting prediction mean: ', np.mean(prediction))
                    print('After adjusting prediction std: ', np.std(prediction))
                    valid_predictions = prediction.flatten()
                    valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                    df = pd.Series(valid_predictions)
                    disggregation_dict[optimizer + '_' + appliance] = df
                    pipeline_util.log_pipeline_run(mode='test', optimizer=optimizer, metrics=self.errors)
        results = pd.DataFrame(disggregation_dict, dtype='float32')
        test_predictions.append(results)

        # It might not have time stamps sometimes due to neural nets
        # It has the readings for all the appliances

        concat_pred_df = pd.concat(test_predictions,axis=0)

        gt = {} # ground truth
        for meter,data in test_submeters:
            concatenated_df_app = pd.concat(data,axis=1)
            index = concatenated_df_app.index
            gt[meter] = pd.Series(concatenated_df_app.values.flatten(),index=index)

        gt_overall = pd.DataFrame(gt, dtype='float32')
        pred = {}

        for app_name in concat_pred_df.columns:
            app_series_values = concat_pred_df[app_name].values.flatten()
            # Neural nets do extra padding sometimes, to fit, so get rid of extra predictions
            app_series_values = app_series_values[:len(gt_overall[app_name.split('_')[-1]])]
            pred[app_name] = pd.Series(app_series_values, index = gt_overall.index)
            print('pred for ', app_name, ' mean: ', pred[app_name])
        pred_overall = pd.DataFrame(pred,dtype='float32')

        return gt_overall, pred_overall


    # metrics
    def compute_loss(self,gt,clf_pred, loss_function):
        error = {}
        for app_name in clf_pred.columns:
            #print('APP NAME: , ', app_name)
            #print('GT: , ', gt[app_name.split('_')[-1]])
            #print('PRED: , ', clf_pred[app_name])
            error[app_name] = loss_function(app_gt=gt[app_name.split('_')[-1]], app_pred=clf_pred[app_name])
        return pd.Series(error)        

    def call_predict(self, timezone):

        """
        This functions computes the predictions on the self.test_mains using all the trained models 
        and then compares different learned models using the metrics specified
        """

        pred_overall={}
        gt_overall={}       # ground truth     
        gt_overall, pred_overall = self.predict(self.test_mains, self.test_submeters, self.sample_period, timezone)

        self.gt_overall=gt_overall
        self.pred_overall=pred_overall
        if gt_overall.size==0:
            print ("No samples found in ground truth")
            return None
        for metric in self.metrics:
            try:
                loss_function = globals()[metric]               
            except:
                print ("Loss function ",metric, " is not supported currently!")
                continue

            computed_metric={}
            computed_metric = self.compute_loss(gt_overall, pred_overall, loss_function)
            computed_metric = pd.DataFrame(computed_metric)
            print("............ " ,metric," ..............")
            print(computed_metric) 
            self.errors.append(computed_metric)
            self.errors_keys.append(self.storing_key + "_" + metric)

        print('HEADS')
        pred_overall.head(5)
        gt_overall.head(5)
        if self.display_predictions:
            for col in pred_overall.columns:
                _make_plots(data={'Truth':gt_overall[col.split('_')[-1]], 'Prediction':pred_overall[col]}, name=col)
                plt.figure(figsize=(100, 12))
        

if __name__ == "__main__":
    nilm_eval().test()
