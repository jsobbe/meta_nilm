import meta
import util

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("problem", "simple", "Type of problem.")

flags.DEFINE_string("path", None, "Path to saved meta-optimizer network.")
flags.DEFINE_string("output_path", None, "Path to output results.")

flags.DEFINE_integer("num_epochs", 1, "Number of evaluation epochs.")
flags.DEFINE_integer("num_steps", 10000, "Number of optimization steps per epoch.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_integer("seed", None, "Seed for TensorFlow's RNG.")


def disaggregate_chunk(self,test_main_list,model=None,do_preprocessing=True):
    if model is not None:
        self.models = model

    # Preprocess the test mains such as windowing and normalizing

    if do_preprocessing:
        test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')

    test_predictions = []
    for test_main in test_main_list:
        test_main = test_main.values
        test_main = test_main.reshape((-1, self.sequence_length, 1))
        disggregation_dict = {}
        for appliance in self.models:
            prediction = self.models[appliance].predict(test_main,batch_size=self.batch_size)
            prediction = self.appliance_params[appliance]['mean'] + prediction * self.appliance_params[appliance]['std']
            valid_predictions = prediction.flatten()
            valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
            df = pd.Series(valid_predictions)
            disggregation_dict[appliance] = df
        results = pd.DataFrame(disggregation_dict, dtype='float32')
        test_predictions.append(results)
    return test_predictions
    
def test(self,d):
    # store the test_main readings for all buildings
    for dataset in d:
        print("Loading data for ",dataset, " dataset")
        test=DataSet(d[dataset]['path'])
        for building in d[dataset]['buildings']:
            test.set_window(start=d[dataset]['buildings'][building]['start_time'],
                            end=d[dataset]['buildings'][building]['end_time'])
            test_mains=next(test.buildings[building].elec.mains().load(physical_quantity='power', 
                                                                       ac_type=self.power['mains'],
                                                                       sample_period=self.sample_period))
            if self.DROP_ALL_NANS and self.site_only:
                test_mains, _= self.dropna(test_mains,[])

            if self.site_only != True:
                appliance_readings=[]

                for appliance in self.appliances:
                    test_df=next((test.buildings[building].elec[appliance].load(
                        physical_quantity='power', ac_type=self.power['appliance'], 
                        sample_period=self.sample_period)))
                    appliance_readings.append(test_df)

                if self.DROP_ALL_NANS:
                    test_mains , appliance_readings = self.dropna(test_mains,appliance_readings)

                if self.artificial_aggregate:
                    print ("Creating an Artificial Aggregate")
                    test_mains = pd.DataFrame(np.zeros(appliance_readings[0].shape),
                                              index = appliance_readings[0].index,
                                              columns=appliance_readings[0].columns)
                    for app_reading in appliance_readings:
                        test_mains+=app_reading
                for i, appliance_name in enumerate(self.appliances):
                    self.test_submeters.append((appliance_name,[appliance_readings[i]]))

            self.test_mains = [test_mains]
            self.storing_key = str(dataset) + "_" + str(building) 
            self.call_predict(self.classifiers, test.metadata["timezone"])
    
    
def predict(self, clf, test_elec, test_submeters, sample_period, timezone ):
    print ("Generating predictions for :",clf.MODEL_NAME)        
    """
    Generates predictions on the test dataset using the specified classifier.
    """

    # "ac_type" varies according to the dataset used. 
    # Make sure to use the correct ac_type before using the default parameters in this code.   
    problem, _, _ = util.get_config(FLAGS.problem, FLAGS.path)
    loss, truth, prediction = problem()
    #TODO run multiple times with the same variables/references? -> Make problem instantiable with constructor and data

    pred_list = clf.disaggregate_chunk(test_elec)

    # It might not have time stamps sometimes due to neural nets
    # It has the readings for all the appliances

    concat_pred_df = pd.concat(pred_list,axis=0)

    gt = {} # ground truth
    for meter,data in test_submeters:
            concatenated_df_app = pd.concat(data,axis=1)
            index = concatenated_df_app.index
            gt[meter] = pd.Series(concatenated_df_app.values.flatten(),index=index)

    gt_overall = pd.DataFrame(gt, dtype='float32')
    pred = {}

    if self.site_only ==True:
        for app_name in concat_pred_df.columns:
            app_series_values = concat_pred_df[app_name].values.flatten()
            pred[app_name] = pd.Series(app_series_values)
        pred_overall = pd.DataFrame(pred,dtype='float32')
        pred_overall.plot(label="Pred")
        plt.title('Disaggregated Data')
        plt.legend()

    else:
        for app_name in concat_pred_df.columns:
            app_series_values = concat_pred_df[app_name].values.flatten()
            # Neural nets do extra padding sometimes, to fit, so get rid of extra predictions
            app_series_values = app_series_values[:len(gt_overall[app_name])]
            pred[app_name] = pd.Series(app_series_values, index = gt_overall.index)
        pred_overall = pd.DataFrame(pred,dtype='float32')

    return gt_overall, pred_overall


# metrics
def compute_loss(self,gt,clf_pred, loss_function):
    error = {}
    for app_name in gt.columns:
        error[app_name] = loss_function(gt[app_name],clf_pred[app_name])
    return pd.Series(error)        
    
def call_predict(self, classifiers, timezone):

    """
    This functions computes the predictions on the self.test_mains using all the trained models and then compares different learned models using the metrics specified
    """

    pred_overall={}
    gt_overall={}       # ground truth     
    for name,clf in classifiers:
        gt_overall,pred_overall[name]=self.predict(clf,self.test_mains,self.test_submeters, 
                                                   self.sample_period, timezone)

    self.gt_overall=gt_overall
    self.pred_overall=pred_overall
    if self.site_only != True:
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
            for clf_name,clf in classifiers:
                computed_metric[clf_name] = 
                self.compute_loss(gt_overall, pred_overall[clf_name], loss_function)
            computed_metric = pd.DataFrame(computed_metric)
            print("............ " ,metric," ..............")
            print(computed_metric) 
            self.errors.append(computed_metric)
            self.errors_keys.append(self.storing_key + "_" + metric)


    if self.display_predictions:
        if self.site_only != True:
            for i in gt_overall.columns:
                plt.figure()
                #plt.plot(self.test_mains[0],label='Mains reading')
                plt.plot(gt_overall[i],label='Truth')
                for clf in pred_overall:                
                    plt.plot(pred_overall[clf][i],label=clf)
                    plt.xticks(rotation=90)
                plt.title(i)
                plt.legend()
                plt.xlabel('Time')
                plt.ylabel('Power (W)')
            plt.show()


def dropna(self,mains_df, appliance_dfs=[]):
    """
    Drops the missing values in the Mains reading and appliance readings and returns consistent data by computing the intersection
    """
    print ("Dropping missing values")

    # The below steps are for making sure that data is consistent by doing intersection across appliances
    mains_df = mains_df.dropna()
    ix = mains_df.index
    mains_df = mains_df.loc[ix]
    for i in range(len(appliance_dfs)):
        appliance_dfs[i] = appliance_dfs[i].dropna()

    for  app_df in appliance_dfs:
        ix = ix.intersection(app_df.index)
    mains_df = mains_df.loc[ix]
    new_appliances_list = []
    for app_df in appliance_dfs:
        new_appliances_list.append(app_df.loc[ix])
    return mains_df,new_appliances_list