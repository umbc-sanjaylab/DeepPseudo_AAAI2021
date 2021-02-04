""" The code is inspired by the code for DeepHit model. The github link of the code for DeepHit is https://github.com/chl8856/DeepHit. Reference: C. Lee, W. R. Zame, J. Yoon, M. van der Schaar, "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks," AAAI Conference on Artificial Intelligence (AAAI), 2018.

   ## Cause-specific Marginal DeepPseudo model's architecture:

Inputs:
    - input_dims: dictionary of dimension information
        > x_dim       : dimension of covariates
        > num_Event   : number of competing events (excluding censoring label)
        > num_Category: dimension of time horizon of interest, i.e., |T| where T = {0, 1, ..., T_max-1}
        > num_evalTime: number of evaluation time points
                   
    - network_settings:
        > num_units_shared    : number of nodes in the shared network
        > num_units_CS        : number of nodes in the cause-specific network
        > num_layers_shared   : number of hidden layers (fully-connected layers) used in the shared network
        > num_layers_CS       : number of hidden layers (fully-connected layers) used in the cause-specific network
        
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


class CS_Marginal_DeepPseudo_Model:
    def __init__(self, sess, name, input_dims, network_settings):
        self.sess               = sess
        self.name               = name

        # INPUT DIMENSIONS
        self.x_dim              = input_dims['x_dim']
        self.num_Event          = input_dims['num_Event']
        self.num_Category       = input_dims['num_Category']
        self.num_evalTime       = input_dims['num_evalTime']

        # NETWORK HYPER-PARMETERS
        self.num_units_shared   = network_settings['num_units_shared']
        self.num_units_CS       = network_settings['num_units_CS']
        self.num_layers_shared  = network_settings['num_layers_shared']
        self.num_layers_CS      = network_settings['num_layers_CS']


        self.activation_fn      = network_settings['activation_fn']
        self.initial_W          = network_settings['initial_W']
        self.reg_W              = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.reg_W_out          = tf.contrib.layers.l1_regularizer(scale=0.1)

        self._build_net()

        
    def _build_net(self):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            ## Placeholder
            self.mb_size    = tf.placeholder(tf.int32, [], name='batch_size')          #Batch Size
            self.lr_rate    = tf.placeholder(tf.float32, [], name='learning_rate')     #Learning Rate
            self.keep_prob  = tf.placeholder(tf.float32, [], name='keep_probability')  #keeping rate: 1- dropout rate

            self.x          = tf.placeholder(tf.float32, shape=[None, self.x_dim], name='inputs') #Covariates
            self.k          = tf.placeholder(tf.float32, shape=[None, 1], name='labels')  #Event/censoring label (censoring:0)
            self.t          = tf.placeholder(tf.float32, shape=[None, 1], name='timetoevents') #Time until event to occur
            self.y1         = tf.placeholder(tf.float32, shape=[None, self.num_evalTime], name='pseudo1') #Pseudo values for CIF for cause 1
            self.y2         = tf.placeholder(tf.float32, shape=[None, self.num_evalTime], name='pseudo2')#pseudo values for CIF for cause 2




            ## Get output from shared network
            shared_out = util_net.create_FCNet(self.x, self.num_layers_shared, self.num_units_shared, self.activation_fn, self.num_units_shared, self.activation_fn, self.initial_W, self.keep_prob, self.reg_W)
            
            ## Get output for cause 1 from cause-specific network feeding output from shared network as input
            cs_out1= util_net.create_FCNet(shared_out, self.num_layers_CS, self.num_units_shared, self.activation_fn, self.num_units_CS, self.activation_fn, self.initial_W, self.keep_prob, self.reg_W)
            self.out1= FC_Net(cs_out1, self.num_evalTime, activation_fn=tf.nn.selu, 
                         weights_initializer=self.initial_W, weights_regularizer=self.reg_W_out, scope="Output1")
            
            
            ## Get output for cause 2 from cause-specific network feeding output from shared network as input
            cs_out2= util_net.create_FCNet(shared_out, self.num_layers_CS, self.num_units_shared, self.activation_fn, self.num_units_CS, self.activation_fn, self.initial_W, self.keep_prob, self.reg_W)
            self.out2= FC_Net(cs_out2, self.num_evalTime, activation_fn=tf.nn.selu, 
                         weights_initializer=self.initial_W, weights_regularizer=self.reg_W_out, scope="Output2")
           

       
            ## Stack the outputs of cause 1 and cause 2 of the event
            out = tf.stack((self.out1, self.out2), axis=1)
           
            ## Reshape outputs
            self.output = tf.reshape(out, [-1, self.num_Event, self.num_evalTime])
            
          

            ## Get loss function
            self.loss_mse_1() 
            self.loss_mse_2()
                 
            ## Optimization
            self.LOSS_TOTAL = self.LOSS1 + self.LOSS2
            self.solver = tf.train.AdamOptimizer(learning_rate=self.lr_rate).minimize(self.LOSS_TOTAL)

    ## Cause 1 specific Loss Function 
    def loss_mse_1(self):
        self.LOSS1 = tf.reduce_mean(tf.squared_difference(self.out1, self.y1))
    
    ## Cause 2 specific Loss Function 
    def loss_mse_2(self):
        self.LOSS2 = tf.reduce_mean(tf.squared_difference(self.out2, self.y2))
    
    ## Cost Function 
    def get_cost(self, DATA, keep_prob, lr_train):
        (x_mb, y1_mb, y2_mb) = DATA
        return self.sess.run(self.LOSS_TOTAL, 
                             feed_dict={self.x:x_mb, self.y1:y1_mb, self.y2:y2_mb,
                                        self.mb_size: np.shape(x_mb)[0], self.keep_prob:keep_prob, self.lr_rate:lr_train})
   
    ## Training Function
    def train(self, DATA,keep_prob, lr_train):
        (x_mb, y1_mb, y2_mb) = DATA
        return self.sess.run([self.solver, self.LOSS_TOTAL], 
                             feed_dict={self.x:x_mb, self.y1:y1_mb, self.y2:y2_mb,
                                        self.mb_size: np.shape(x_mb)[0], self.keep_prob:keep_prob, self.lr_rate:lr_train})
    ## Prediction Function
    def predict(self, x_test, keep_prob=1.0):
        return self.sess.run(self.output, feed_dict={self.x: x_test, self.mb_size: np.shape(x_test)[0], self.keep_prob: keep_prob})

 