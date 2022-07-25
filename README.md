# meta_nilm

This repository contains the code for my master's thesis 'Learning to Learn Neural Networks for Energy Disaggregation'.
The implementation uses adapted code from both the [Open-L2O](https://github.com/VITA-Group/Open-L2O) and the [NILM-TK](https://github.com/nilmtk/nilmtk).
It can be used to reproduce the experimental results from the thesis or run custom experiments.


## Setup

1. Create a conda environment based on the `environment.yml` so that all dependencies are installed.
2. In order to run any experiments, a NILM dataset is required. Data is expected to be in the `/data` directory in the `.h5` format. Datasets (we used [REDD](http://redd.csail.mit.edu/), [iAWE](https://iawe.github.io/) and [UK-DALE](https://jack-kelly.com/data/)) can be converted to `.h5` using the NILMTK, which is included in this repository. Instructions on loading and converting the data can be found in its [documentation](https://github.com/nilmtk/nilmtk/tree/master/docs/manual).


## Usage

### Pipeline
The pipeline for running experiments consists of three main stages: 

1. __Meta-training__  
Here the meta learner is trained by repeatedly training a NILM model from scratch and learning a good optimization strategy or function for the problem. The training data for the meta-learner is the loss produced by the base-learner in each iteration, which the meta-learner tries to minimize by adjusting its own weights. For executing this stage, there are two entrypoints depending on the employed optimizer:  
DM optimizer:		`python ./meta/train_dm.py > ./meta/logs/0_train/dm.out`      
RNNprop:		`python ./meta/train_rnnprop.py > ./meta/logs/0_train/rnn.out`
    
2. __Meta-evaluation__  
Here a number of optimizers are evaluated by training a new S2P network for a given amount of steps. The optimizers can either be classic ones or previously trained meta-learners.  
`python ./meta/evaluate.py > ./meta/logs/1_eval/eval.out`

3. __NILM-evaluation__  
At this stage, the trained NILM base-learner is run on the test dataset and its predictions are evaluated using different established metrics for energy disaggregation.  
`python ./meta/eval_nilm.py  > ./meta/logs/2_nilm/nilm.out`

### Configuration
Each stage has its own configuration file which allows setting all parameters for the respective experiments. On top of that there is also a configuration file for the data which centrally controls the data used for the experiments.

* __`conf_train`__ This config file allows to define a trainings run for the meta learner. It contains options for output files, for the appliances to train on and for specifying a name under which to save the model and results. The training itself can be customized by setting various parameters such as number of epochs and steps, unroll length etc, or by enabling imitation and curriculum learning techniques. Some specific models and techniques also have their own available parameters.
* __`conf_eval`__ Defines all relevant parameters for the meta-evaluation stage. It contains a section for managing the output files for the experiments. There is another section for the general setup including number of steps and epochs and the seeds to use for each run.
* __`conf_nilm`__ This configuration contains both some central configuration for the S2P network, that should remain the same across all stages, and the configuration for the NILM-evaluation stage. For the latter one can specify the output files, the appliances, for which to conduct the evaluation, and the optimizers and metrics to use. For the S2P network there are general options such as window and batch size and options for the preprocessing.
* __`conf_data`__ This file simply contains the dataset definition for each stage. It defines which houses from which dataset are to be used as well as the timespan for each house.



## Results

### Logs
The designated location of the logs is in `meta/logs/` but needs to be specified as the output stream when running an experiment (see pipeline for examples).

### Generating plots
All results achieved during the meta evaluation stage can be plotted using the `reproduce_plots.ipynb`.
All results achieved during the meta evaluation stage can be analysed using the `analyse_results.ipynb`.


### Reproducibility
Some notes on the reproducibility of our results: During the meta evaluation stage, for each run a seed can be defined in advance, that is then set as a random seed by numpy and tensorflow for any kind of random number generation. This should in theory ensure reproducible results, but unfortunately we found that this does not quite hold in practice. We cannot say for sure why this happens, but there are several potential sources for non-determinism in tensorflow, some of which are listed [here](https://github.com/NVIDIA/framework-determinism/blob/master/doc/tensorflow.md). As many of the tensorflow libraries used in the Open-L2O are sparsely (some not at all) documented legacy versions, it is difficult to tell whether these issues originate from one of them. From our experience though the results are reliably similar enough to back our findings, when rerunning the experiments with the same parameters and the provided models.

### Issues
In some cases during optimizer training, the network only returns `NaN`s as loss. Rerunning the experiment usually rectifies that problem.

