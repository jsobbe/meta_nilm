from nilmtk.api import API
from nilmtk.contrib.disaggregate import Seq2Point

import warnings
warnings.filterwarnings("ignore")

class nilm_test():

    def __init__(self):
        self.experiment = {
          'power': {'mains': ['apparent'], 'appliance': ['active']},
          'sample_rate': 3,
          'appliances': ['fridge'],
          'methods': {"Seq2Point":{} },
          'train': {    
            'datasets': {
                'redd':{
                    'path': './data/redd.h5',
                    'buildings': {
                        1: {'start_time': '2011-04-01', 'end_time': '2011-05-10'}
                    }}}
            },
          'test': {
            'datasets': {
               'redd':{
                    'path': './data/redd.h5',
                    'buildings': {
                        1: {'start_time': '2011-05-11', 'end_time': '2011-05-30'}
                    }}
                },
                'metrics':['mae']
            }
        }
        
    def run():
        api_results_experiment_1 = API(experiment1) 
        
    
if __name__ == "__main__":
    nilm_test().run()   
    