############################ IMPORTS #########################

#standard ds imports
import pandas as pd
import numpy as np
import os

#visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

#import custom modules
from env import user, password, host

#import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#ignore warnings
import warnings
warnings.filterwarnings("ignore")


############################## AQUIRE ZILLOW FUNCTION ##############################

def acquire_zillow():
    '''
    This function checks to see if zillow.csv already exists, 
    if it does not, one is created
    '''
    #check to see if telco_churn.csv already exist
    if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)
    
    else:

        #creates new csv if one does not already exist
        df = get_zillow_data()
        df.to_csv('zillow.csv')

    return df

############################## REMOVE OUTLIERS FUNCTION ##############################

def remove_outliers(df, k, col_list):
    ''' 
    This function takes in a dataframe, the threshold and a list of columns 
    and returns the dataframe with outliers removed
    '''   
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

############################ PREPARE ZILLOW FUNCTION ###########################

def prep_zillow(df):
    '''
    This function takes in the zillow df
    then the data is cleaned and returned
    '''
    #change column names to be more readable
    df = df.rename(columns={'bedroomcnt':'bedrooms','bathroomcnt':'bathrooms', 'calculatedfinishedsquarefeet':'sqft', 'taxvaluedollarcnt':'home_value', 'taxamount':'sale_tax', 'yearbuilt':'year_built'})

    #drop null values- at most there were 9000 nulls (this is only 0.5% of 2.1M)
    df = df.dropna()

    #drop duplicates
    df.drop_duplicates(inplace=True)
    
    #removing outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'sqft', 'home_value', 'sale_tax'])
   
    return df


############################ WRANGLE ZILLOW FUNCTION ############################

def wrangle_zillow():
    '''
    This function acquires and prepares our Zillow data
    and returns the clean dataframe
    '''
    df = prep_zillow(acquire_zillow())
    return df



############################ SPLIT ZILLOW FUNCTION ############################

def split_zillow(df):
    '''
    This function takes in the dataframe
    and splits it into train, validate, test datasets
    '''    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test


########################################## AQUIRE ZILLOW FUNCTION #######################################

def split_clean_zillow():
    '''
    This function splits our clean dataset into 
    train, validate, test datasets
    '''
    train, validate, test = split_zillow(wrangle_zillow())
    
    print(f"train: {train.shape}")
    print(f"validate: {validate.shape}")
    print(f"test: {test.shape}")
    
    return train, validate, test
        


