from __future__ import print_function, division
from warnings import warn
from nilmtk.disaggregate import Disaggregator
import pandas as pd
import numpy as np
from collections import OrderedDict 
import matplotlib.pyplot as  plt
from sklearn.decomposition import MiniBatchDictionaryLearning, SparseCoder
from sklearn.metrics import mean_squared_error
import time
import warnings
warnings.filterwarnings("ignore")

class DSC(Disaggregator):
    
    def __init__(self, params):

        self.MODEL_NAME = 'DSC'  # Add the name for the algorithm
        self.chunk_wise_training = False
        self.dictionaries = OrderedDict()
        self.power = OrderedDict()
        self.shape = 60*2
        self.learning_rate = 1e-9
        self.iterations = 3000
        self.sparsity_coef = 20
        self.n_components = 10
        self.shape = params.get('shape',self.shape)
        self.learning_rate = params.get('learning_rate',self.learning_rate)
        self.iterations = params.get('iterations',self.iterations)
        self.n_epochs = self.iterations
        self.n_components = params.get('n_components',self.n_components)

    def learn_dictionary(self, appliance_main, app_name):

        if appliance_main.size%self.shape!=0:
            extra_values = self.shape - (appliance_main.size)%(self.shape)
            appliance_main = list(appliance_main.values.flatten()) + [0]*extra_values
        appliance_main = np.array(appliance_main).reshape((-1,self.shape)).T
        self.power[app_name] = appliance_main

        if app_name not in self.dictionaries:
            print ("Training First dictionary for ",app_name)
            model = MiniBatchDictionaryLearning(n_components=self.n_components,positive_code=True,positive_dict=True,transform_algorithm='lasso_lars',alpha=self.sparsity_coef)
        
        else:
            print ("Re-training dictionary for ",app_name)
            model = self.dictionaries[app_name]
        model.fit(appliance_main.T)
        reconstruction = np.matmul(model.components_.T,model.transform(appliance_main.T).T)
        print ("RMSE reconstruction for appliance %s is %s"%(app_name,mean_squared_error(reconstruction,appliance_main)**(.5)))
        self.dictionaries[app_name] = model
        

    def discriminative_training(self,concatenated_activations,concatenated_bases, verbose = 100):


        # Making copies of concatenated bases and activation. 
        optimal_a = np.copy(concatenated_activations)
        predicted_b = np.copy(concatenated_bases)
        
        '''
        Next step is to modify bases such that, we get optimal A upon sparse coding
        We want to get a_opt on finding activations from b_hat
        '''

        alpha = self.learning_rate
        least_error = 1e10
        total_power = self.total_power
        v_size = .20
        v_index = int(total_power.shape[1] * v_size)
        train_power = total_power[:,:-v_index]
        v_power = total_power[:,-v_index:]
        train_optimal_a = optimal_a[:,:-v_index]
        v_optimal_a = optimal_a[:,-v_index:]

        print ("If Iteration wise errors are not decreasing, then please decrease the learning rate")
        for i in range(self.iterations):

            a = time.time()
            # Finding activations for the given bases
            model = SparseCoder(dictionary=predicted_b.T,positive_code=True,transform_algorithm='lasso_lars',transform_alpha=self.sparsity_coef)
            train_predicted_a = model.transform(train_power.T).T
            model = SparseCoder(dictionary=predicted_b.T,positive_code=True,transform_algorithm='lasso_lars',transform_alpha=self.sparsity_coef)
            val_predicted_a = model.transform(v_power.T).T        
            err = np.mean(np.abs(val_predicted_a - v_optimal_a))

            if err<least_error:
                #print ("Chose the best")
                least_error = err
                best_b = np.copy(predicted_b)
                
            # Modify the bases b_hat so that they result activations closer to a_opt
            T1 = (train_power - predicted_b@train_predicted_a)@train_predicted_a.T
            T2 = (train_power - predicted_b@train_optimal_a)@train_optimal_a.T
            predicted_b = predicted_b - alpha *( T1 - T2)
            predicted_b = np.where(predicted_b>0,predicted_b,0)
            # Making sure that columns sum to 1
            predicted_b = (predicted_b.T/np.linalg.norm(predicted_b.T,axis=1).reshape((-1,1))).T 
            #if i%verbose==0:
            print ("Iteration ",i," Error ",err)

        return  best_b

    def print_appliance_wise_errors(self, activations, bases):

        start_comp = 0        
        for cnt, i in enumerate(self.power):
            X = self.power[i]
            n_comps = self.dictionaries[i].n_components
            pred = np.matmul(bases[:,start_comp:start_comp+n_comps],activations[start_comp:start_comp+n_comps,:])
            start_comp+=n_comps
            #plt.plot(pred.T[home_id],label=i)
            print ("Error for ",i," is ",mean_squared_error(pred, X)**(.5))
        
    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        
        print("...............DSC partial_fit running...............")

        #print (train_main[0])

        train_main = pd.concat(train_main,axis=1) #np.array([i.values.reshape((self.sequence_length,1)) for i in train_main])
        
        if train_main.size%self.shape!=0:
            extra_values = self.shape - (train_main.size)%(self.shape)
            train_main = list(train_main.values.flatten()) + [0]*extra_values
        
        train_main = np.array(train_main).reshape((-1,self.shape)).T
        self.total_power = train_main
        new_train_appliances  = []
        
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df)
            new_train_appliances.append((app_name, app_df))
            
        train_appliances = new_train_appliances
        
        if len(train_main)>10:

            for appliance_name, power in train_appliances:
                self.learn_dictionary(power, appliance_name)

            concatenated_bases = []
            concatenated_activations = []

            for i in self.dictionaries:
                
                model = self.dictionaries[i]
                
                concatenated_bases.append(model.components_.T)
                concatenated_activations.append(model.transform(self.power[i].T).T)

            concatenated_bases = np.concatenate(concatenated_bases,axis=1)
            concatenated_activations = np.concatenate(concatenated_activations,axis=0)
            print ("--"*15)
            print ("Optimal Errors")
            self.print_appliance_wise_errors(concatenated_activations, concatenated_bases)
            print ("--"*15)
            model = SparseCoder(dictionary=concatenated_bases.T,positive_code=True,transform_algorithm='lasso_lars',transform_alpha=self.sparsity_coef)
            predicted_activations = model.transform(train_main.T).T
            print ('\n\n')
            print ("--"*15)
            print ("Error in prediction before discriminative sparse coding")
            self.print_appliance_wise_errors(predicted_activations, concatenated_bases)
            print ("--"*15)
            print ('\n\n')
            optimal_b = self.discriminative_training(concatenated_activations,concatenated_bases)
            model = SparseCoder(dictionary=optimal_b.T,positive_code=True,transform_algorithm='lasso_lars',transform_alpha=self.sparsity_coef)
            self.disggregation_model = model
            predicted_activations = model.transform(train_main.T).T
            print ("--"*15)
            print ("Model Errors after Discriminative Training")
            self.print_appliance_wise_errors(predicted_activations, concatenated_bases)
            print ("--"*15)
            self.disaggregation_bases = optimal_b
            self.reconstruction_bases = concatenated_bases
            
        else:
            print ("This chunk has small number of samples, so skipping the training")

    def disaggregate_chunk(self, test_main_list):

        test_predictions = []
        for test_main in test_main_list:
            if test_main.size%self.shape!=0:
                extra_values = self.shape - (test_main.size)%(self.shape)
                test_main = list(test_main.values.flatten()) + [0]*extra_values
            test_main = np.array(test_main).reshape((-1,self.shape)).T
            predicted_activations = self.disggregation_model.transform(test_main.T).T
            #predicted_usage = self.reconstruction_bases@predicted_activations
            disggregation_dict = {}
            start_comp = 0            
            for cnt, app_name in enumerate(self.power):
                n_comps = self.dictionaries[app_name].n_components
                predicted_usage = np.matmul(self.reconstruction_bases[:,start_comp:start_comp+n_comps],predicted_activations[start_comp:start_comp+n_comps,:])
                start_comp+=n_comps
                predicted_usage = predicted_usage.T.flatten() 
                flat_mains = test_main.T.flatten()
                predicted_usage = np.where(predicted_usage>flat_mains,flat_mains,predicted_usage)
                disggregation_dict[app_name] = pd.Series(predicted_usage)
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)

        return test_predictions

