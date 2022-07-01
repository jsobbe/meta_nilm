# meta_nilm

This repository contains the code for my master's thesis 'Learning to Learn Neural Networks for Energy Disaggregation'.

It can be used to reproduce the experimental results.


## Setup

1. Create a conda environment based on the environment.yml so that all dependencies are installed.
2. In order to run any experiments, a NILM dataset is required. Data is expected to be in the /data directory in the .h5 format. Datasets (e.g. [REDD](http://redd.csail.mit.edu/)) can be converted to .h5 using the NILMTK, which is included in this repository. Instructions on loading and converting the data can be found in its [documentation](https://github.com/nilmtk/nilmtk/tree/master/docs/manual).


## Usage

The pipeline for running experiments consists of three main stages: 

* __Meta-training__ Here the meta learner is trained by repeatedly training a nilm model from scratch and learning a good optimization strategy or function for the problem. The training data for the meta-learner is the loss produced by the base-learner in each iteration, which the meta-learner tries to minimize by adjusting its own weights.\\
	During the training stage, the meta optimizers are trained for up to 1000 epochs. During each epoch they are given a new unseen S2P network that they need to optimize. The S2P is presented with 400 batches of labelled data and based on the mean squared error that it returns the optimizer can adjust its own weights and thus its update rule. At the end of the training, the optimizer network with the best validation result is stored for later evaluation.
* __Meta-evaluation__ Here a number of optimizers are evaluated by training a new S2P network for a given amount of steps. The optimizers can either be classic ones or previously trained meta-learners. All optimizers here are evaluated using the same pre-defined seed, to assure that their results are actually comparable. This process can be repeated for a number of times in order to obtain an average of multiple runs and thus providing more robust and diverse results. The resulting network of the optimizee is then saved for a detailed evaluation of its capabilities for energy disaggregation.
* __Nilm-evaluation__ At this stage, the trained NILM base-learner is run on the test dataset and its predictions are evaluated using different established metrics for energy disaggregation.

Each stage has its own configuration file which allows setting all parameters for the respective experiment.
