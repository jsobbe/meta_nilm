DATASETS_TRAIN = {
#     'redd':{
#         'path': './data/redd.h5',
#             'buildings': {
#                 2: {'start_time': '2011-04-01', 'end_time': '2011-04-30'}, 
#                 3: {'start_time': '2011-04-01', 'end_time': '2011-04-30'}, 
#                 4: {'start_time': '2011-04-01', 'end_time': '2011-04-30'}, 
#                 5: {'start_time': '2011-04-01', 'end_time': '2011-04-30'}, 
#                 6: {'start_time': '2011-04-01', 'end_time': '2011-04-30'}
#     }}, 
    'UKDALE':{
        'path': './data/ukdale.h5',
            'buildings': {
                1: {'start_time': '2013-06-01', 'end_time': '2013-06-30'} ,
                2: {'start_time': '2013-06-01', 'end_time': '2013-06-30'} ,
                3: {'start_time': '2013-03-01', 'end_time': '2013-03-30'},
                4: {'start_time': '2013-06-01', 'end_time': '2013-06-30'}
    }}
                 }



# DATASETS_EVAL = {'iAWE':{
#     'path': './data/iawe.h5',
#     'buildings': {
#         1: {'start_time': '2013-07-13', 'end_time': '2013-08-04'} 
#     }}}
DATASETS_EVAL = {
    'redd':{
        'path': './data/redd.h5',
            'buildings': {
                2: {'start_time': '2011-05-01', 'end_time': '2011-05-28'}, 
                3: {'start_time': '2011-05-01', 'end_time': '2011-05-28'}, 
                4: {'start_time': '2011-05-01', 'end_time': '2011-05-28'}, 
                5: {'start_time': '2011-05-01', 'end_time': '2011-05-28'}, 
                6: {'start_time': '2011-05-01', 'end_time': '2011-05-28'}
    }}}



DATASETS_TEST = {
    'redd':{
        'path': './data/redd.h5',
            'buildings': {
                1: {'start_time': '2011-04-01', 'end_time': '2011-05-31'}
    }}}
# DATASETS_TEST = {'UKDALE':{
#     'path': './data/ukdale.h5',
#     'buildings': {
#         1: {'start_time': '2012-12-01', 'end_time': '2013-04-01'} 
#     }}}
# DATASETS_TEST = {'iAWE':{
#     'path': './data/iawe.h5',
#     'buildings': {
#         1: {'start_time': '2013-07-13', 'end_time': '2013-08-04'} 
#     }}}