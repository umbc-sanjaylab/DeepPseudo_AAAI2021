""" The code is inspired by the code for DeepHit model. The github link of the code for DeepHit is https://github.com/chl8856/DeepHit. Reference: C. Lee, W. R. Zame, J. Yoon, M. van der Schaar, "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks," AAAI Conference on Artificial Intelligence (AAAI), 2018.

This 'get_validation_performance.py' trains the Cause-specific Marginal DeepPseudo model and give the validation C-index performance for random search.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
from termcolor import colored

from import_data import import_data
from CS_Marginal_DeepPseudo import CS_Marginal_DeepPseudo_Model
from utils_eval import c_index, brier_score, weighted_c_index, weighted_brier_score


'tensorflow =1.14.0'
'numpy      =1.16.5'
'pandas     =0.24.2'

##### USER-DEFINED FUNCTIONS
def log(x):
    return tf.log(x + 1e-8)


def div(x, y):
    return tf.div(x, (y + 1e-8))


def f_get_minibatch(mb_size, x, y1, y2):
    """Get minibatches.
    Arguments:
      - mb_size: size of the minibatch
      - x: covariates
      - y1: pseudo values for CIF for cause 1
      - y2: pseudo values for CIF for cause 2
    Returns:
      - minibatches of covariates and pseudo values
    
    """
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb = x[idx, :].astype(np.float32)
    y1_mb = y1[idx, :].astype(np.float32) 
    y2_mb = y2[idx, :].astype(np.float32)
    return x_mb, y1_mb, y2_mb


def get_valid_performance(in_parser, out_itr, evalTime=None, MAX_VALUE = -99, OUT_ITERATION=5):
    """ Trains the Marginal DeepPseudo model and give the validation C-index performance for random search.

    Arguments:
        - in_parser: dictionary of hyperparameters
        - out_itr: indicator of set of 5-fold cross validation datasets
        - evalTime: None or a list(e.g. [12, 60]). Evaluation times at which the validation performance is measured
        - MAX_VALUE: maximum validation value
        - OUT_ITERATION: Total number of the set of cross-validation data

    Returns:
        - the validation performance of the trained network
        - save the trained network in the folder directed by "in_parser['out_path'] + '/itr_' + str(out_itr)"
    """
    
    ## Define a list of continuous columns from the covariates
    continuous_columns=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11','feature12']
    ## If there are categorical variables in the covariates, define a list of the categorical variables
    
    ## Import the attributes 
    tr_data, tr_time, tr_label, y_train, va_data, va_time, va_label, y_val, te_data, te_time, te_label, y_test, num_Category, num_Event, num_evalTime, x_dim = import_data(out_itr, evalTime, categorical_columns=None, continuous_columns=continuous_columns)
    y_train1 = y_train[:,0,:] #pseudo values for CIF for cause 1
    y_train2 = y_train[:,1,:] #pseudo values for CIF for cause 2
    
    ## Hyper-parameters
    ACTIVATION_FN               = {'selu': tf.nn.selu, 'elu': tf.nn.elu, 'tanh': tf.nn.tanh, 'relu':tf.nn.relu}    
    mb_size                     = in_parser['mb_size']
    iteration                   = in_parser['iteration']
    keep_prob                   = in_parser['keep_prob']
    lr_train                    = in_parser['lr_train']
    initial_W                   = tf.contrib.layers.xavier_initializer()


    ## Make Dictionaries
    # Input Dimensions
    input_dims                  = { 'x_dim'         : x_dim,
                                    'num_Event'     : num_Event,
                                    'num_Category'  : num_Category,
                                    'num_evalTime'  : len(evalTime)}

    # NETWORK HYPER-PARMETERS
    network_settings        = { 'num_units_shared'   : in_parser['num_units_shared'],
                                'num_layers_shared'  : in_parser['num_layers_shared'],
                                'num_units_CS'       : in_parser['num_units_CS'],
                                'num_layers_CS'      : in_parser['num_layers_CS'],
                                'activation_fn'      : ACTIVATION_FN[in_parser['activation_fn']],
                                'initial_W'          : initial_W }


    file_path_final = in_parser['out_path'] + '/itr_' + str(out_itr)

    #change parameters...
    if not os.path.exists(file_path_final + '/models/'):
        os.makedirs(file_path_final + '/models/')


    ## Use GPU
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    ## Call the Marginal DeepPseudo Model
    model = CS_Marginal_DeepPseudo_Model(sess, "CS_Marginal_DeepPseudo", input_dims, network_settings)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    
    
    
    max_valid = -99
    stop_flag = 0

 

    ### Training - Main
    print( "MAIN TRAINING ...")
    print( "EVALUATION TIMES: " + str(evalTime))

    avg_loss = 0
    for itr in range(iteration):
        if stop_flag > 10: #for faster early stopping
            break
        else:
            x_mb, y1_mb, y2_mb= f_get_minibatch(mb_size, tr_data, y_train1, y_train2)   #get the minibatches
            DATA = (x_mb, y1_mb, y2_mb)
            _, loss_curr = model.train(DATA, keep_prob, lr_train)                       #train the model
            avg_loss += loss_curr/1000
                
            if (itr+1)%1000 == 0:
                print('|| ITR: ' + str('%04d' % (itr + 1)) + ' | Loss: ' + colored(str('%.4f' %(avg_loss)), 'yellow' , attrs=['bold']))
                avg_loss = 0

            ### Validation based on the average C-index
            if (itr+1)%1000 == 0:
                
                ### Prediction for validation data
                pred = model.predict(va_data)
                

                ### Evaluation on validation data
                val_result = np.zeros([num_Event, len(evalTime)])

                for t, t_time in enumerate(evalTime):
                    eval_horizon = int(t_time)
                    if eval_horizon >= num_Category:
                        print('ERROR: evaluation horizon is out of range')
                        val_result[:, t]  = -1
                    else:
                        risk = pred[:,:,t]                     #risk score until evalTime
                        for k in range(num_Event):
                            val_result[k, t] = weighted_c_index(tr_time, (tr_label[:,0] == k+1).astype(int), risk[:,k], va_time, (va_label[:,0] == k+1).astype(int), eval_horizon)  #weighted c-index calculation for validation data
                                                  
                tmp_valid = np.mean(val_result)    #average weighted C-index

                if tmp_valid >  max_valid:
                    stop_flag = 0
                    max_valid = tmp_valid
                    print( 'updated.... average c-index = ' + str('%.4f' %(tmp_valid)))

                    if max_valid > MAX_VALUE:
                        saver.save(sess, file_path_final + '/models/model_itr_' + str(out_itr))
                else:
                    stop_flag += 1

    return max_valid

