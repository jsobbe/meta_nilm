# meta_nilm

This repository contains the code for my master's thesis 'Learning to Learn Neural Networks for Energy Disaggregation'.

It can be used to reproduce the experimental results.


## Setup

1. Create a conda environment based on the environment.yml so that all dependencies are installed.
2. In order to run any experiments, a NILM dataset is required. Data is expected to be in the /data directory in the .h5 format. Datasets (we used [REDD](http://redd.csail.mit.edu/), [iAWE](https://iawe.github.io/) and [UK-DALE](https://jack-kelly.com/data/)) can be converted to .h5 using the NILMTK, which is included in this repository. Instructions on loading and converting the data can be found in its [documentation](https://github.com/nilmtk/nilmtk/tree/master/docs/manual).


## Usage

The pipeline for running experiments consists of three main stages: 

vvvvv __TODO__ rework this  vvvvv  taken from the thesis vvvvv

* __Meta-training__ Here the meta learner is trained by repeatedly training a nilm model from scratch and learning a good optimization strategy or function for the problem. The training data for the meta-learner is the loss produced by the base-learner in each iteration, which the meta-learner tries to minimize by adjusting its own weights.\\
	During the training stage, the meta optimizers are trained for up to 1000 epochs. During each epoch they are given a new unseen S2P network that they need to optimize. The S2P is presented with 400 batches of labelled data and based on the mean squared error that it returns the optimizer can adjust its own weights and thus its update rule. At the end of the training, the optimizer network with the best validation result is stored for later evaluation.
* __Meta-evaluation__ Here a number of optimizers are evaluated by training a new S2P network for a given amount of steps. The optimizers can either be classic ones or previously trained meta-learners. All optimizers here are evaluated using the same pre-defined seed, to assure that their results are actually comparable. This process can be repeated for a number of times in order to obtain an average of multiple runs and thus providing more robust and diverse results. The resulting network of the optimizee is then saved for a detailed evaluation of its capabilities for energy disaggregation.
* __NILM-evaluation__ At this stage, the trained NILM base-learner is run on the test dataset and its predictions are evaluated using different established metrics for energy disaggregation.

^^^^^ __TODO__ rework this ^^^^^ taken from the thesis ^^^^^

Each stage has its own configuration file which allows setting all parameters for the respective experiments. On top of that there is a configuration file for the data which centrally controls the data used for the experiements.

* __conf_train__ This config file allows to define a trainings run for the meta learner. It contains options for output files, the appliances to train on and specifying a name under which to save the model and results. The training itself can be customized by setting various parameters such as number of epochs and steps, unroll length etc. or enabling imitation and curriculum learning techniques. Some specific models and techniques also have their own available parameters.
* __conf_eval__ Defines all relevant parameters for the meta-evaluation stage. It contains a section for managing the output files for the experiments. One section for the general setup including number of steps and epochs and the seeds to use for each run. 
* __conf_nilm__ This configuration contains both some central configuration for the S2P network, that should remain the same across all stages, and the configuration for the NILM-evaluation stage. For the latter the output files, the appliances for which to conduct the evaluation and the optimizers and metrics to use. For the S2P network there are general options such as window and batch size and options for the preprocessing.
* __conf_data__ This file simply contains the dataset definition for each stage. It defines which houses from which dataset are to be used as well as the timespan for each house.

## Results

#### Logs
The designated location of the logs is in `meta/logs/` but needs to be specified when running an experiment as the output stream.

#### Generating plots
All results achieved during the meta evaluation stage can be plotted using the `reproduce_plots.ipynb`.


### Reproducibility
Some notes on the reproducibility of our results: During the meta evaluation stage, for each run a seed can be defined in advance, that is then set as a random seed by numpy and tensorflow for any kind of random number generation. This should in theory ensure reproducible results, but unfortunately we found that this does not quite hold in practice. We cannot say for sure why this happens, but there are several potential sources for non-determinism in tensorflow, some of which are listed [here](https://github.com/NVIDIA/framework-determinism/blob/master/doc/tensorflow.md). As many of the tensorflow libraries used in the Open-L2O are sparsely (some not at all) documented legacy versions, it is difficult to tell whether these issues originate from one of them. From our experience though the results are reliably similar enough to back our findings, when rerunning the experiments with the same parameters and the provided models.



