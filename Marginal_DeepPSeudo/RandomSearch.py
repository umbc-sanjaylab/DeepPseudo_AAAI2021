""" The code is inspired by the code for DeepHit model. The github link of the code for DeepHit is https://github.com/chl8856/DeepHit. Reference: C. Lee, W. R. Zame, J. Yoon, M. van der Schaar, "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks," AAAI Conference on Artificial Intelligence (AAAI), 2018.

This 'RandomSearch.py' runs random search to find the optimal hyper-parameters setting

Inputs:
    - OUT_ITERATION: total number of the set of cross-validation data
    - RS_ITERATION : number of random search iteration
    - evalTime: None or a list(e.g. [12, 60]). Evaluation times at which the validation performance is measured

Outputs:
    - "hyperparameters_log.txt" is the output
    - Once the hyper parameters are optimized, run "summarize_results.py" to get the final results.
"""
import os
import time as tm
import get_validation_performance
import numpy as np
import argparse
import copy
import utils


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", default='.')
    return parser.parse_args()


# this function saves the current hyperparameters
def save_logging(dictionary, log_name):
    with open(log_name, 'w') as f:
        for key, value in dictionary.items():
            f.write('%s:%s\n' % (key, value))


# this function call the saved hyperparameters
def load_logging(filename):
    data = dict()
    with open(filename) as f:
        def is_float(input):
            try:
                num = float(input)
            except ValueError:
                return False
            return True

        for line in f.readlines():
            if ':' in line:
                key,value = line.strip().split(':', 1)
                if value.isdigit():
                    data[key] = int(value)
                elif is_float(value):
                    data[key] = float(value)
                elif value == 'None':
                    data[key] = None
                else:
                    data[key] = value
            else:
                pass # deal with bad lines of text here    
    return data


# this function randomly select hyperparamters based on the given list of candidates
def get_random_hyperparameters(out_path, iteration):
    SET_BATCH_SIZE    = [64, 128, 512]               #size of minibatches
    SET_LAYERS        = [2,3,4,5]                    #number of layers
    SET_NODES         = [32, 64, 128, 512]           #number of nodes
    SET_LR            = [0.001, 0.0001, 0.00001]     #learning rate

    new_parser = {'mb_size'   : SET_BATCH_SIZE[np.random.randint(len(SET_BATCH_SIZE))],
                 'iteration'  : iteration,
                 'keep_prob'  : 0.6,
                 'lr_train'   : SET_LR[np.random.randint(len(SET_LR))],
                 'num_units'  : SET_NODES[np.random.randint(len(SET_NODES))],
                 'num_layers' : SET_LAYERS[np.random.randint(len(SET_LAYERS))],
                 'activation_fn'  : 'selu',
                 'out_path'   :out_path}
    
    return new_parser #outputs the dictionary of the randomly-chosen hyperparamters



import argparse
import pandas as pd


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o")
    parser.add_argument("--it", default=100000, type=int)  #number of iterations. default is 100000
    parser.add_argument("--itout", default=5, type=int)    #total number of the set of cross-validation data
    parser.add_argument("--itrs", default=30, type=int)     #number of random searches. default is 30.
    return parser.parse_args()



args = init_arg()
odir = 'Synthetic'    #output directory
logger = utils.init_logger(odir, 'log_marginal_deeppseudo.txt')
data_mode='Synthetic'
##Main settings
OUT_ITERATION               = args.itout
RS_ITERATION                = args.itrs
logger.info('data_mode:{}'.format(data_mode))
iteration = args.it


#Evaluation Times
evalTime = [12, 60] 

out_path      = odir + '/results/'


for itr in range(OUT_ITERATION):    
    if not os.path.exists(out_path + '/itr_' + str(itr) + '/'):
        os.makedirs(out_path + '/itr_' + str(itr) + '/')

    max_valid = 0.
    log_name = out_path + '/itr_' + str(itr) + '/hyperparameters_log.txt'

    for r_itr in range(RS_ITERATION):
        time_start_iter = tm.time()
        logger.info('OUTER_ITERATION: ' + str(itr+1) + '({})'.format(OUT_ITERATION))
        logger.info('Random search... itr: ' + str(r_itr+1) + '({})'.format(RS_ITERATION))
        new_parser = get_random_hyperparameters(out_path, iteration)
        logger.info('{}'.format(new_parser))

        # get validation performance given the hyperparameters
        tmp_max = get_validation_performance.get_valid_performance(new_parser, itr, evalTime, MAX_VALUE=max_valid)

        if tmp_max > max_valid:
            max_valid = tmp_max
            max_parser = new_parser
            save_logging(max_parser, log_name)  #save the hyperparameters if this provides the maximum validation performance

        logger.info('Current best: ' + str(max_valid))
        logger.info('time iter:{}s'.format(tm.time() - time_start_iter))
        
