import pandas as pd
import os.path
from pandas import datetime
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re


from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
from sklearn.model_selection import learning_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier

def pca_date_parser(x):
    return parse_date(x, '%d.%m.%Y')

def macro_date_parser(x):
    return parse_date(x, '%Y-%m-%d')

def parse_date(dt, pattern):
    return datetime.strptime(str(dt), pattern)

#Query to get data
#select timeseries_id, date, value, sname from timeseries_value join timeseries on timeseries_value.timeseries_id = timeseries.id
#where timeseries.sname in ('Poland All industries 1-2 Credit Curve (EUR)',
#'Poland All industries 1-2 Credit Curve (PLN)',
#'Singapore All industries 1-2 Credit Curve (SGD)')

df = pd.read_csv('C:/Users/98250/Documents/R/Russell_check_y.csv', parse_dates=['date'])
id_unique = df['id'].unique()
len(id_unique)

window_size = 20 # usually, range 10-20, the size of moving window, if the window wide you can catch broad outliers, 
#but usually they are tiny, so window should be narrow

all_filtered_data = pd.DataFrame(columns=['id','date', 'value','sname'])


def get_median_filtered(signal, threshold = 10): #range 10-60 #show how many difference between signal 
#and median in this window (divided on median of this difference) is bigger of some threshold 
    """
    signal: is numpy array-like
    returns: signal, numpy array 
    """
    difference = np.abs(signal - np.median(signal))
    median_difference = np.median(difference)
    s = 0 if median_difference == 0 else difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal
        
for ts in id_unique:
    current_frame =  df.loc[df['id'] == ts]
    value = current_frame.value.astype(float)
    value_s = value.tolist()
    median_filtered_signal = []
    for ii in range(0, value.size, window_size): 
        median_filtered_signal += get_median_filtered(np.asanyarray(value_s[ii: ii+window_size])).tolist() 
#        median_filtered_signal= median_filtered_signal.append(get_median_filtered(np.asanyarray(value[ii: ii+window_size])))
    value = pd.Series(median_filtered_signal)
    
    date = pd.to_datetime(current_frame['date'])  
    value = pd.DataFrame(value).set_index(date.index.values)
    value.columns = ['value']
    sname = current_frame['sname']
    
#    filtered_data['date'] = date
    filtered_data = pd.concat([date,value,sname], axis =1)
    filtered_data['id'] = ts

    all_filtered_data = all_filtered_data.append(filtered_data)

outliers = df.loc[df['value'] != all_filtered_data['value']]
outliers_fix = all_filtered_data.loc[df['value'] != all_filtered_data['value']]

outliers_fix.to_csv('C:/Users/98250/Documents/R/outliers20_10_Russell_check_y.csv')   
all_filtered_data.to_csv('C:/Users/98250/Documents/R/median20_10_Russell_check_y.csv')
 
total = pd.concat([df, all_filtered_data], axis = 1)
total.to_csv('C:/Users/98250/Documents/R/total20_10_Russell_check_y.csv')  
