## Import Packages
import numpy as np
import pandas as pd

## Import "rpy2" to run the "R" script in Python
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from rpy2.robjects.vectors import IntVector, FloatVector
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import Converter, localconverter
import rpy2.robjects as robjects


## Function for getting pseudo values
def pseudoFunction(data, evaltime):
    """Calculate the pseudo values for cumulative incidence function.
    Arguments:
        data: A dataframe containing survival time and event status, which will be used to estimate pseudo values for CIF.
        evaltime: Evaluation times at which pseudo values are calcuated
    Returns:
        A dataframe of pseudo values for all subjects for all causes at the evaluation time points. 
    """
    r=robjects.r
    r.source('Compute_pseudo_values.r')
    ps=r.pseudo(data, evaltime)
    return ps


## Function to convert categorical variables into one-hot-encoded dummy variables
def to_one_hot(dataframe, columns):
    """Convert columns in dataframe to one-hot encoding.
    Arguments:
        dataframe (dataframe): pandas dataframe containing covariates
        columns (list of strings): list categorical column names to one hot encode
    Returns:
        one_hot_df (dataframe): dataframe with categorical columns encoded as binary variables
    """

    
    one_hot_df = pd.get_dummies(data=dataframe, columns=columns)
    
    return one_hot_df

## Function to standardize the continuous variables
def standardized(train_df, test_df, continuous_columns):
    """Standardize the continuous columns in dataframe.
    Arguments:
        train_df: training dataframe containing covariates
        test_df: test dataframe whose continuous columns will be standardized
        continuous_columns: A list of name of the continuous columns in the dataframe
    Returns:
        A new test dataframe whose continuous columns ared standardized
    """
    mean = train_df.loc[:, continuous_columns].mean()
    stdev = train_df.loc[:, continuous_columns].std()
    test_df.loc[:, continuous_columns] = (test_df.loc[:, continuous_columns] - mean) / (stdev+1e-8)
    return test_df



def import_data(out_itr, evalTime, categorical_columns=None, continuous_columns=None):
    """Preprocess the data to use them to the model to train, validate and predict
     Arguments: 
        out_itr: indicator of set of 5-fold cross validation data out of 5 simulated dataset
        evalTime: Evaluation times 
        categorical_columns: A list of name of the categorical columns in the dataframe
        continuous_columns: A list of name of the continuous columns in the dataframe
    Returns:
        All the attributes that will be used in the model to train, validate and predict
    """

    ### Loading Data from the folder named as the dataset (Synthetic/WIHS/SEER) in the code directory
    train_df = pd.read_csv('Synthetic/train_data_{}.csv'.format(out_itr))
    val_df = pd.read_csv('Synthetic/valid_data_{}.csv'.format(out_itr))
    test_df = pd.read_csv('Synthetic/test_data_{}.csv'.format(out_itr))
    
    
    #Create a column 'train' to trainining, validation and test data and combined them. Then convert the the categorical variables into dummy variables on combined data so that the number of columns in all three dataset remain equal. 
    train_df['train']=1
    val_df['train']=2
    test_df['train']=3
    df=pd.concat([train_df, val_df, test_df])
    
    #Convert the categorical variables into dummy variables
    if categorical_columns is not None:
        df = to_one_hot(df, categorical_columns)
    train_data=df[df['train']==1]    
    val_data=df[df['train']==2]
    test_data=df[df['train']==3]
    
    #Drop the 'train' column from all three datasets.
    train_data=train_data.drop(columns=['train'])
    val_data=val_data.drop(columns=['train'])
    test_data=test_data.drop(columns=['train'])
    
    #Standardize the contunuous columns
    if continuous_columns is not None:
        train_data=standardized(train_data, train_data, continuous_columns)
        val_data=standardized(train_data, val_data, continuous_columns)
        test_data=standardized(train_data, test_data, continuous_columns)
    
    #Full Dataset
    dataset     = df.drop(columns=['train'])
    label       = np.asarray(dataset[['status']])
    time        = np.asarray(dataset[['time']])
    data        = np.asarray(dataset.drop(columns=['status', 'time']))


    num_Category    = int(np.max(time) * 1.2)  #to have enough time-horizon
    num_Event       = int(len(np.unique(label)) - 1) #the number of events (excluding censoring as an event)
    x_dim           = np.shape(data)[1]  #No. of covarites in the dataset
    num_evalTime    = len(evalTime)      #No. of evaluation times
    
    
    #Preprocess the Training Data
    train_feat=train_data.drop(columns=['time','status']) 
    tr_data = np.asarray(train_feat)
    tr_time=np.asarray(train_data[['time']])
    tr_label=np.asarray(train_data[['status']])
    eval_time=FloatVector(evalTime)
    #Convert the 'Python' dataframe to 'R'
    with localconverter(default_converter + pandas2ri.converter) as cv:
        train_data_pseudo = pandas2ri.py2ri(train_data)
    train_pseudo=pseudoFunction(train_data_pseudo, eval_time)  
    y_train=np.reshape(train_pseudo, [-1, num_Event, len(evalTime)])

    

    #Preprocess the Validation Data
    val_feat=val_data.drop(columns=['time','status'])
    va_data = np.asarray(val_feat)
    va_time=np.asarray(val_data[['time']])
    va_label=np.asarray(val_data[['status']])
    #Convert the 'Python' dataframe to 'R'
    with localconverter(default_converter + pandas2ri.converter) as cv:
        val_data_pseudo = pandas2ri.py2ri(val_data)
    val_pseudo=pseudoFunction(val_data_pseudo, eval_time)  
    y_val=np.reshape(val_pseudo, [-1, num_Event, len(evalTime)])

       
   

    #Preprocess the Test Data
    test_feat=test_data.drop(columns=['time','status'])
    te_data = np.asarray(test_feat)
    te_time=np.asarray(test_data[['time']])
    te_label=np.asarray(test_data[['status']])
    #Convert the 'Python' dataframe to 'R'
    with localconverter(default_converter + pandas2ri.converter) as cv:
        test_data_pseudo = pandas2ri.py2ri(test_data)
    test_pseudo=pseudoFunction(test_data_pseudo, eval_time)  
    y_test=np.reshape(test_pseudo, [-1, num_Event, len(evalTime)])
       
    
    
    return tr_data, tr_time, tr_label, y_train, va_data, va_time, va_label, y_val, te_data, te_time, te_label, y_test, num_Category, num_Event, num_evalTime, x_dim
