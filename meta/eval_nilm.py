import meta
import util
import nilm_config
import nilm_seq2point
import tensorflow as tf

from nilmtk.dataset import DataSet
from nilmtk.losses import *
from nilmtk.metergroup import MeterGroup
import pandas as pd
from nilmtk.losses import *
import numpy as np
import matplotlib.pyplot as plt
import datetime

def _dropna(mains_df, appliance_dfs=[]):
    """
    Drops the missing values in the Mains reading and appliance readings and returns consistent data by computing the intersection
    """
    # The below steps are for making sure that data is consistent by doing intersection across appliances
    mains_df = mains_df.dropna()
    ix = mains_df.index
    mains_df = mains_df.loc[ix]
    for i in range(len(appliance_dfs)):
        appliance_dfs[i] = appliance_dfs[i].dropna()

        
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

class nilm_eval():
    
    def __init__(self, problem):
        self.metrics = nilm_config.METRICS
        self.appliances = nilm_config.APPLIANCES
        self.drop_nans = nilm_config.DROP_NANS
        self.power = nilm_config.POWER
        self.appliances = nilm_config.APPLIANCES
        self.window_size = nilm_config.WINDOW_SIZE
        self.sample_period = nilm_config.SAMPLE_PERIOD
        self.batch_size = nilm_config.BATCH_SIZE
        self.artificial_aggregate = nilm_config.ARTIFICIAL_AGGREGATE
        self.test_submeters = []
        self.errors = []
        self.errors_keys = []
        self.do_preprocessing = nilm_config.PREPROCESSING
        self.display_predictions = nilm_config.DISPLAY_PRED
        self.optimizers = nilm_config.OPTIMIZERS

    def test(self):
        # store the test_main readings for all buildings
        d = nilm_config.DATASETS_TEST

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

                if self.drop_nans:
                    test_mains, appliance_readings = _dropna(test_mains,appliance_readings)
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
#         print ("Generating predictions for :",clf.MODEL_NAME)        
        """
        Generates predictions on the test dataset using the specified classifier.
        """

        # "ac_type" varies according to the dataset used. 
        # Make sure to use the correct ac_type before using the default parameters in this code.   
        #TODO run multiple times with the same variables/references? -> Make problem instantiable with constructor and data
        window_size = nilm_config.WINDOW_SIZE
        
        if self.do_preprocessing:
            test_main_list = nilm_seq2point.call_preprocessing(test_elec, submeters_lst=None, method='nilm_test', window_size=window_size)

        test_predictions = []
        for test_main in test_main_list:
            mains_len = len(test_main)
            test_main = test_main.values
            test_main = test_main.reshape((-1, window_size, 1))
            test_main = tf.squeeze(tf.convert_to_tensor(test_main))
            disggregation_dict = {}
            for optimizer in self.optimizers:
                for appliance in self.appliances:
                    with tf.Session() as sess:
                        problem = nilm_seq2point.model(mode='nilm_test', mains=test_main, mains_len=mains_len, load=True, optimizer=optimizer, appliance_name=appliance)()
                        sess.run(tf.global_variables_initializer())
                        sess.run(tf.local_variables_initializer())
                        prediction = sess.run(problem)
                        prediction = self.appliance_params[appliance]['mean'] + prediction * self.appliance_params[appliance]['std'] #TODO make sure mean is calculated correctly in advance!
#                         print('#1 Prediction mean: ', np.mean(prediction))
#                         print('#1 Prediction std: ', np.std(prediction))
                        valid_predictions = prediction.flatten()
                        valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                        df = pd.Series(valid_predictions)
                        disggregation_dict[optimizer + '_' + appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)

            #print('MADE PREDICTION:')
            #print('type: ', type(test_predictions))
            #print('len: ', len(test_predictions))
            #print('content: ', str(test_predictions))
        # TODO create nilm_seq2point.get_mains_and_subs_test for new method

        # It might not have time stamps sometimes due to neural nets
        # It has the readings for all the appliances

        concat_pred_df = pd.concat(test_predictions,axis=0)

        gt = {} # ground truth
        for meter,data in test_submeters:
            print('Test sub meter:', type(meter))
            print('Test sub data:', type(data))
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

        print('gt type: ', type(gt_overall))
        print('gt: ', gt_overall)
        print('pred type: ', type(pred_overall))
        print('pred: ', pred_overall)
        return gt_overall, pred_overall


    # metrics
    def compute_loss(self,gt,clf_pred, loss_function):
        error = {}
        for app_name in clf_pred.columns:
            print('APP NAME: , ', app_name)
            print('GT: , ', gt[app_name.split('_')[-1]])
            print('PRED: , ', clf_pred[app_name])
            error[app_name] = loss_function(app_gt=gt[app_name.split('_')[-1]], app_pred=clf_pred[app_name])
        return pd.Series(error)        

    def call_predict(self, timezone):

        """
        This functions computes the predictions on the self.test_mains using all the trained models and then compares different learned models using the metrics specified
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


        if self.display_predictions:
            for i in pred_overall.columns:
                plt.figure()
                #plt.plot(self.test_mains[0],label='Mains reading')
                plt.plot(gt_overall[i.split('_')[-1]],label='Truth')
                plt.plot(pred_overall[i],label='Pred')
                plt.xticks(rotation=90)
                plt.title(i)
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('Power (W)')
                plt.yscale("log")
                plt.savefig('./nilm_results/' + i + '.png')
            plt.show(block=True)

if __name__ == "__main__":
    nilm_eval(None).test()
