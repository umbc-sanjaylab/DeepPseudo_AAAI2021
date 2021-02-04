import numpy as np
import pandas as pd

'numpy      =1.16.5'
'pandas     =0.24.2'

""" First create a folder in the code directory and name it as the dataset name, e.g., Synthetic/SEER/WIHS. Then upload the raw data in the folder. The following function will create 5 sets of 5-fold cross-validation competing risk datasets from raw dataset and the datasets will be saved the in mentioned folder. 
"""
def create_cross_val_data(seed=1):
    
    """Create 5 sets of 5-fold cross-validation competing risk data. We consider 2 causes of the event here.
    
    Arguments:
      seed: An integer used by the random number generator. Default value is set as 1.
      
    Returns:
      5 sets of 5-fold cross-validation competing risk datasets
    """
    
    data=pd.read_csv('Synthetic/synthetic_data.csv')  # Load raw dataset from the folder named as the dataset (Synthetic/WIHS/SEER) in the code directory
    data=data.sample(frac=1, random_state=seed)
    censored_data = data[data['status']==0]           # Extract censored data 
    uncensored_data_1 = data[data['status']==1]       # Extract uncensored data for cause1
    uncensored_data_2 = data[data['status']==2]       # Extract uncensored data for cause2
    
    #Size of the data
    n_cen=censored_data.shape[0]
    n_uncen_1=uncensored_data_1.shape[0]
    n_uncen_2=uncensored_data_2.shape[0]
    
    #Create 5 equal folds from censored data
    cen_fold1=censored_data.iloc[:int(n_cen/5), :]
    cen_fold2=censored_data.iloc[int(n_cen/5): int(n_cen/5)*2, :]
    cen_fold3=censored_data.iloc[int(n_cen/5)*2: int(n_cen/5)*3, :]
    cen_fold4=censored_data.iloc[int(n_cen/5)*3: int(n_cen/5)*4, :]
    cen_fold5=censored_data.iloc[int(n_cen/5)*4:, :]
    
    #Create 5 equal folds from the uncensored data for cause 1
    uncen_1_fold1=uncensored_data_1.iloc[:int(n_uncen_1/5), :]
    uncen_1_fold2=uncensored_data_1.iloc[int(n_uncen_1/5): int(n_uncen_1/5)*2, :]
    uncen_1_fold3=uncensored_data_1.iloc[int(n_uncen_1/5)*2: int(n_uncen_1/5)*3, :]
    uncen_1_fold4=uncensored_data_1.iloc[int(n_uncen_1/5)*3: int(n_uncen_1/5)*4, :]
    uncen_1_fold5=uncensored_data_1.iloc[int(n_uncen_1/5)*4:, :]
    
    #Create 5 equal folds from the uncensored data for cause 2
    uncen_2_fold1=uncensored_data_2.iloc[:int(n_uncen_2/5), :]
    uncen_2_fold2=uncensored_data_2.iloc[int(n_uncen_2/5): int(n_uncen_2/5)*2, :]
    uncen_2_fold3=uncensored_data_2.iloc[int(n_uncen_2/5)*2: int(n_uncen_2/5)*3, :]
    uncen_2_fold4=uncensored_data_2.iloc[int(n_uncen_2/5)*3: int(n_uncen_2/5)*4, :]
    uncen_2_fold5=uncensored_data_2.iloc[int(n_uncen_2/5)*4:, :]
    
    #Concatenated censored and uncensored data and create final 5-folds
    fold1=pd.concat([cen_fold1, uncen_1_fold1, uncen_2_fold1])
    fold1=fold1.sample(frac=1, random_state=seed)
    fold2=pd.concat([cen_fold2, uncen_1_fold2, uncen_2_fold2])
    fold2=fold2.sample(frac=1, random_state=seed)
    fold3=pd.concat([cen_fold3, uncen_1_fold3, uncen_2_fold3])
    fold3=fold3.sample(frac=1, random_state=seed)
    fold4=pd.concat([cen_fold4, uncen_1_fold4, uncen_2_fold4])
    fold4=fold4.sample(frac=1, random_state=seed)
    fold5=pd.concat([cen_fold5, uncen_1_fold5, uncen_2_fold5])
    fold5=fold5.sample(frac=1, random_state=seed)
    
    
    #Create 5 sets of 5-fold cross-validation datasets taking different combinations of folds
    train_1= pd.concat([fold1, fold2, fold3])
    train_2= pd.concat([fold1, fold2, fold5])
    train_3= pd.concat([fold1, fold4, fold5])
    train_4= pd.concat([fold3, fold4, fold5])
    train_5= pd.concat([fold2, fold3, fold4])
    
    valid_1=fold4
    valid_2=fold3
    valid_3=fold2
    valid_4=fold1
    valid_5=fold5
    
    
    test_1=fold5
    test_2=fold4
    test_3=fold3
    test_4=fold2
    test_5=fold1
    
    
    
    #Save training datasets in the folder named as the dataset
    train_1.to_csv('Synthetic/train_data_0.csv', index=False)
    train_2.to_csv('Synthetic/train_data_1.csv', index=False)
    train_3.to_csv('Synthetic/train_data_2.csv', index=False)
    train_4.to_csv('Synthetic/train_data_3.csv', index=False)
    train_5.to_csv('Synthetic/train_data_4.csv', index=False)
    
            
    
    #Save validation datasets in the folder named as the dataset   
    valid_1.to_csv('Synthetic/valid_data_0.csv', index=False)
    valid_2.to_csv('Synthetic/valid_data_1.csv', index=False)
    valid_3.to_csv('Synthetic/valid_data_2.csv', index=False)
    valid_4.to_csv('Synthetic/valid_data_3.csv', index=False)
    valid_5.to_csv('Synthetic/valid_data_4.csv', index=False)
    
    
    
    #Save test datasets in the folder named as the dataset
    test_1.to_csv('Synthetic/test_data_0.csv', index=False)
    test_2.to_csv('Synthetic/test_data_1.csv', index=False)
    test_3.to_csv('Synthetic/test_data_2.csv', index=False)
    test_4.to_csv('Synthetic/test_data_3.csv', index=False)
    test_5.to_csv('Synthetic/test_data_4.csv', index=False)
   
    return

create_cross_val_data(seed=1)