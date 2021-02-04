""" The code is inspired by the code for DeepHit model. The github link of the code for DeepHit is https://github.com/chl8856/DeepHit. Reference: C. Lee, W. R. Zame, J. Yoon, M. van der Schaar, "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks," AAAI Conference on Artificial Intelligence (AAAI), 2018.

   ## Conditional DeepPseudo model's architecture:

Inputs:
    - input_dims: dictionary of dimension information
        > x_dim       : dimension of covariates
        > num_Event   : number of competing events (excluding censoring label)
        > num_Category: dimension of time horizon of interest, i.e., |T| where T = {0, 1, ..., T_max-1}
        > num_evalTime: number of evaluation time points
                   
    - network_settings:
        > num_units    : number of nodes
        > num_layers   : number of hidden layers (fully-connected layers) used in the model
        > activation_fn: 'selu'
        > initial_W    : Xavier initialization

Loss Function:
    - Mean squared error loss. Pseudo values for the cumulative incidence function is used as output variable.
"""

import numpy as np
import tensorflow as tf
import random

from tensorflow.contrib.layers import fully_connected as FC_Net
import utils_network as util_net


'tensorflow =1.14.0'
'numpy      =1.16.5'



##### USER-DEFINED FUNCTIONS
def log(x):
    return tf.log(x + 1e-08)

def div(x, y):
    return tf.div(x, (y + 1e-08))


class Conditional_DeepPseudo_Model:
    def __init__(self, sess, name, input_dims, network_settings):
        self.sess               = sess
        self.name               = name

        # Input dimensions
        self.x_dim              = input_dims['x_dim']
        self.num_Event          = input_dims['num_Event']
        self.num_Category       = input_dims['num_Category']
        self.num_evalTime       = input_dims['num_evalTime']

        # Network hyper-paramters
        self.num_units          = network_settings['num_units']
        self.num_layers         = network_settings['num_layers']


        self.activation_fn      = network_settings['activation_fn']
        self.initial_W          = network_settings['initial_W']
        self.reg_W              = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.reg_W_out          = tf.contrib.layers.l1_regularizer(scale=0.1)

        self._build_net()



    def _build_net(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            ## Placeholder  
            self.mb_size    = tf.placeholder(tf.int32, [], name='batch_size')           #Batch Size
            self.lr_rate    = tf.placeholder(tf.float32, [], name='learning_rate')      #Learning Rate
            self.keep_prob  = tf.placeholder(tf.float32, [], name='keep_probability')   #Keeping rate: 1- dropout rate

            self.x           = tf.placeholder(tf.float32, shape=[None, self.x_dim], name='inputs')#Covariates
            self.k           = tf.placeholder(tf.float32, shape=[None, 1], name='labels')         #event/censoring label (censoring:0)
            self.t           = tf.placeholder(tf.float32, shape=[None, 1], name='timetoevents')   #Time until event to occur
            self.y           = tf.placeholder(tf.float32, shape=[None, 1], name='pseudo')         #Pseudo values for CIF




            ## Get output from final hidden layer
            out_fc = util_net.create_FCNet(self.x, self.num_layers, self.num_units, self.activation_fn, self.num_units, self.activation_fn, self.initial_W, self.keep_prob, self.reg_W)
            
            self.output = FC_Net(out_fc, 1, activation_fn=tf.nn.selu, 
                         weights_initializer=self.initial_W, weights_regularizer=self.reg_W_out, scope="Output")
            
          
            ## Get loss function
            self.loss_mse()      
                 
            ## Optimization
            self.solver = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.LOSS)

    
    
    ## Loss Function
    def loss_mse(self):
        self.LOSS = tf.reduce_mean(tf.squared_difference(self.output, self.y))

    ## Cost Function  
    def get_cost(self, DATA, keep_prob, lr_train):
        (x_mb, y_mb) = DATA
        return self.sess.run(self.LOSS, 
                             feed_dict={self.x:x_mb, self.y:y_mb,
                                        self.mb_size: np.shape(x_mb)[0], self.keep_prob:keep_prob, self.lr_rate:lr_train})

    ## Training Function
    def train(self, DATA,keep_prob, lr_train):
        (x_mb, y_mb) = DATA
        return self.sess.run([self.solver, self.LOSS], 
                             feed_dict={self.x:x_mb, self.y:y_mb, 
                                        self.mb_size: np.shape(x_mb)[0], self.keep_prob:keep_prob, self.lr_rate:lr_train})
    
    ## Prediction Function
    def predict(self, x_test, keep_prob=1.0):
        return self.sess.run(self.output, feed_dict={self.x: x_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})

