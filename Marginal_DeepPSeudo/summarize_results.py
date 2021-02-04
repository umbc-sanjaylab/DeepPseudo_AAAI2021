''' The code is inspired by the code for DeepHit model. The github link of the code for DeepHit is https://github.com/chl8856/DeepHit. Reference: C. Lee, W. R. Zame, J. Yoon, M. van der Schaar, "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks," AAAI Conference on Artificial Intelligence (AAAI), 2018.

This 'summarize_results.py' gives the final outputs of Marginal DeepPseudo model on test data.
'''



import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
from termcolor import colored
from import_data import import_data
from Marginal_DeepPseudo import Marginal_DeepPseudo_Model
from utils_eval import c_index, brier_score, weighted_c_index, weighted_brier_score
import argparse
import utils

'tensorflow =1.14.0'
'numpy      =1.16.5'
'pandas     =0.24.2'

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


def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", default='.')
    parser.add_argument("--itout", default=5, type=int)
    return parser.parse_args()



##### Main settings
args = init_arg()
odir = 'Synthetic'
logger = utils.init_logger(odir, 'log_marginal_deeppseudo_sum.txt')
OUT_ITERATION               = args.itout
data_mode                   = 'Synthetic'
num_Event                   =  2        #causes of the event
evalTime                    = [12, 60]  # evalution times (for C-index and Brier-Score)
in_path = odir + '/results/'


if not os.path.exists(in_path):
    os.makedirs(in_path)



WEIGHTED_C_INDEX  = np.zeros([num_Event, len(evalTime), OUT_ITERATION])
WEIGHTED_BRIER_SCORE = np.zeros([num_Event, len(evalTime), OUT_ITERATION])



for out_itr in range(OUT_ITERATION):
    
    ## Define a list of continuous columns from the covariates
    continuous_columns=['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11','feature12']
    ## If there are categorical variables in the covariates, define a list of the categorical variables
    
    
    ## Import the attributes
    tr_data, tr_time, tr_label, y_train, va_data, va_time, va_label, y_val, te_data, te_time, te_label, y_test, num_Category, num_Event, num_evalTime, x_dim = import_data(out_itr, evalTime, categorical_columns=None, continuous_columns=continuous_columns)

    in_hypfile = in_path + '/itr_' + str(out_itr) + '/hyperparameters_log.txt'
    in_parser = load_logging(in_hypfile)


    ## Hyper-parameters
    mb_size                     = in_parser['mb_size']
    iteration                   = in_parser['iteration']
    keep_prob                   = in_parser['keep_prob']
    lr_train                    = in_parser['lr_train']
    num_units                   = in_parser['num_units']
    num_layers                  = in_parser['num_layers']

    if in_parser['activation_fn']   == 'selu':
        activation_fn                   = tf.nn.selu
    elif in_parser['activation_fn'] == 'elu':
        activation_fn                   = tf.nn.elu
    elif in_parser['activation_fn'] == 'tanh':
        activation_fn                   = tf.nn.tanh
    elif in_parser['activation_fn'] == 'relu':
        activation_fn                   = tf.nn.relu
    else:
        print('Error!')
        
    initial_W                   = tf.contrib.layers.xavier_initializer()



    ## Make Dictionaries
    # Input Dimensions
    input_dims                  = { 'x_dim'         : x_dim,
                                    'num_Event'     : num_Event,
                                    'num_Category'  : num_Category,
                                    'num_evalTime'  : len(evalTime)}

    # Network hyper-paramters
    network_settings            = { 'num_units'     : num_units,
                                    'num_layers'    : num_layers,
                                    'activation_fn' : activation_fn,
                                    'initial_W'     : initial_W }




    ## Use GPU
    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    ## Call the Marginal DeepPseudo Model
    model = Marginal_DeepPseudo_Model(sess, "Marginal_DeepPseudo", input_dims, network_settings)
    print('number of trainable parameter:',np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    
    ## Prediction & Evaluation
    saver.restore(sess, in_path + '/itr_' + str(out_itr) + '/models/model_itr_' + str(out_itr))

    ### Prediction for test data
    pred = model.predict(te_data)
    pred_br=np.minimum(1, np.maximum(0, pred))  #clipping transformation for brier score calculation
     

    ### Evaluation on test data
    result1, result2 = np.zeros([num_Event, len(evalTime)]), np.zeros([num_Event, len(evalTime)])

    for t, t_time in enumerate(evalTime):
        eval_horizon = int(t_time)
        if eval_horizon >= num_Category:
            print( 'ERROR: evaluation horizon is out of range')
            result1[:, t] = result2[:, t] = -1
        else:
            risk = pred[:,:,t]     #risk score until evalTime
            risk_br=pred_br[:,:,t]
            
            for k in range(num_Event):
                result1[k, t] = weighted_c_index(tr_time, (tr_label[:,0] == k+1).astype(int), risk[:,k], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                result2[k, t] = weighted_brier_score(tr_time, (tr_label[:,0] == k+1).astype(int), risk_br[:,k], te_time, (te_label[:,0] == k+1).astype(int), eval_horizon) #-1 for no event (not comparable)
                
    WEIGHTED_C_INDEX[:, :, out_itr]     = result1
    WEIGHTED_BRIER_SCORE[:, :, out_itr] = result2


    ### SAVE RESULTS
    row_header = []
    for t in range(num_Event):
        row_header.append('Event_' + str(t+1))

    col_header1 = []
    col_header2 = []
    for t in evalTime:
        col_header1.append(str(t) + 'months')
        col_header2.append(str(t) + 'months')


    # weighted c-index result
    df1 = pd.DataFrame(result1, index = row_header, columns=col_header1)
    df1.to_csv(in_path + '/result_weighted_CINDEX_itr' + str(out_itr) + '.csv')

    # weighted brier-score result
    df2 = pd.DataFrame(result2, index = row_header, columns=col_header2)
    df2.to_csv(in_path + '/result_weighted_BRIER_itr' + str(out_itr) + '.csv')
    


    ### PRINT RESULTS
    print('========================================================')
    print('ITR: ' + str(out_itr+1))
    print('Parameters: ' + 'num_units = '+str(num_units) + ' num_layers = '+str(num_layers) + ' Learning Rate: ' + str(lr_train))
   

    print('========================================================')
    print('- Weighted C-INDEX: ')
    print(df1)
    print('--------------------------------------------------------')
    print('- Weighted BRIER-SCORE: ')
    print(df2)
    print('========================================================')


    
### FINAL MEAN/STD
# weighted c-index result
df1_mean = pd.DataFrame(np.mean(WEIGHTED_C_INDEX, axis=2), index = row_header, columns=col_header1)
df1_std  = pd.DataFrame(np.std(WEIGHTED_C_INDEX, axis=2), index = row_header, columns=col_header1)
df1_mean.to_csv(in_path + '/result_WEIGHTED_CINDEX_FINAL_MEAN.csv')
df1_std.to_csv(in_path + '/result_WEIGHTED_CINDEX_FINAL_STD.csv')

# Weighted brier-score result
df2_mean = pd.DataFrame(np.mean(WEIGHTED_BRIER_SCORE, axis=2), index = row_header, columns=col_header2)
df2_std  = pd.DataFrame(np.std(WEIGHTED_BRIER_SCORE, axis=2), index = row_header, columns=col_header2)
df2_mean.to_csv(in_path + '/result_WEIGHTED_BRIER_FINAL_MEAN.csv')
df2_std.to_csv(in_path + '/result_WEIGHTED_BRIER_FINAL_STD.csv')



### PRINT RESULTS
print('========================================================')
print('- FINAL WEIGHTED C-INDEX: ')
print(df1_mean)
print('--------------------------------------------------------')
print('- FINAL WEIGHTED BRIER-SCORE: ')
print(df2_mean)
print('========================================================')

