#evaluate.py

#import ignore warninings
import warnings
warnings.filterwarnings("ignore")

#standard ds imports
import pandas as pd
import numpy as np
from pydataset import data

#visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

#math imports
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import explained_variance_score

#####################FUNCTION TO PLOT RESIDUALS - HISTOGRAM #####################

def plot_residuals_hist(df, y, yhat):
    '''
    This function takes in actual value and predicted value 
    then creates a historgram of the residuals
    '''
    # residuals
    df['residuals'] = df[y] - df[yhat]
    
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.hist(df.residuals, label='model residuals', alpha=.6)
    ax.legend()
    return

#####################FUNCTION TO PLOT RESIDUALS#####################

def plot_residuals(y, yhat):
    '''
    This function takes in actual value and predicted value 
    then creates a scatter plot of those values
    '''
    residuals = y - yhat
    
    plt.scatter(x=y, y=residuals)
    plt.xlabel('Home Value')
    plt.ylabel('Residuals')
    plt.title('Residual vs Home Value Plot')
    plt.show()
    
#####################FUNCTION FOR REGRESSION ERRORS#####################

def regression_errors(y, yhat):
    '''
    This function takes in actual value and predicted value 
    then outputs: the sse, ess, tss, mse, and rmse
    '''
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = math.sqrt(MSE)
    ESS = ((yhat - y.mean())**2).sum()
    TSS = ESS + SSE
    
    return MSE, SSE, RMSE, ESS, TSS

#####################FUNCTION FOR REGRESSION ERRORS WITH PRINT STATEMENT#####################

def regression_errors_print(y, yhat):
    '''
    This function takes in actual value and predicted value 
    then outputs a print statement of sse, ess, tss, mse, and rmse
    '''
    MSE = mean_squared_error(y, yhat)
    SSE = MSE * len(y)
    RMSE = math.sqrt(MSE)
    ESS = ((yhat - y.mean())**2).sum()
    TSS = ESS + SSE
        
    print(f''' 
        SSE: {SSE: .4f}
        ESS: {ESS: .4f}
        TSS: {TSS: .4f}
        MSE: {MSE: .4f}
        RMSE: {RMSE: .4f}
        ''')

#####################FUNCTION FOR BASELINE ERRORS#####################

def baseline_mean_errors(y):
    '''
    This function takes in actual value and predicted value
    then outputs: the SSE, MSE, and RMSE for the baseline model
    '''
    baseline = np.repeat(y.mean(), len(y))
    
    MSE = mean_squared_error(y, baseline)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    return MSE, SSE, RMSE

#####################FUNCTION FOR BASELINE ERRORS WITH PRINT STATEMENT#####################

def baseline_mean_errors_print(y):
    '''
    This function takes in actual value and predicted value
    then outputsa print statement of the SSE, MSE, and RMSE for the baseline
    '''
    baseline = np.repeat(y.mean(), len(y))
    
    MSE = mean_squared_error(y, baseline)
    SSE = MSE * len(y)
    RMSE = MSE**.5
    
    print(f'''
        sse_baseline: {SSE: .4f}
        mse_baseline: {MSE: .4f}
        rmse_baseline: {RMSE: .4f}
        ''')

##################FUNCTION TO RETURN BETTER THAN BASELINE##################

def better_than_baseline(y, yhat):
    '''
    This function takes in the target and the prediction
    then returns a print statement 
    to inform us if the model outperforms the baseline
    '''
    SSE, ESS, TSS, MSE, RMSE = regression_errors(y, yhat)
    
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
    
    if SSE < SSE_baseline:
        print('My OSL model performs better than baseline')
    else:
        print('My OSL model performs worse than baseline. :( )')