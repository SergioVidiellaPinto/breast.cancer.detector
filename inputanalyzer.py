# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import config


def prepare_input(missing_data="drop", test_size=0.2, random=0, 
                  binarize_output=False, normalize=False):
    """Prepare input data for training model"""
    
    # Read the data and generate a matrix of integers and NaN values
    dataset = read_input_data()
    
    # Handle the missing data
    # Separate the input and output values
    if missing_data == "drop":
        dataset.dropna(inplace=True)
        X = dataset.iloc[:, 1:10].values
        X = X.astype(float)
        y = dataset.iloc[:, 10].values
    elif missing_data == "mean":
        imputer = Imputer(missing_values="NaN", strategy='mean', axis=1)
        imputer = imputer.fit(dataset)
        dataset = imputer.transform(dataset)
        X = dataset[:, 1:10]
        y = dataset[:, 10]
    
    if binarize_output:
        y[y == int(config.get_value("data", "benign_value"))] = 0
        y[y == int(config.get_value("data", "malignant_value"))] = 1
        
    # Split the dataset in training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size,
                                                        random_state = random)
    if normalize:
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)        
        
    return X_train, X_test, y_train, y_test


def read_input_data():
    """Read the input file and return the dataset"""
    
    input_file = "{}{}".format(os.path.dirname(os.path.realpath(__file__)),
                               config.get_value("paths", "data_path"))
    dataset = pd.read_csv(input_file,
                          sep=config.get_value("general", "separator"))
    dataset = dataset.replace(config.get_value("general", "no_data"), np.nan)
    
    return dataset


def analyze_column(column):
    """Given a column value between 0 and 9, return the statistics for all 
    values"""
    
    # Only column values between 0 and 9 are accepted
    if column < 0 or column > 9:
        return None
    
    dataset = read_input_data()
    X = dataset.iloc[:, 1:10].values
    X = X.astype(float)
    
    result = {"value_nan": X[:, column].tolist().count(np.nan)}
    
    for i in range(int(config.get_value("data", "min_value")),
                   int(config.get_value("data", "max_value")) + 1):
        result["value_{}".format(i)] =  X[:, column].tolist().count(i)
        
    return result
    

def analyze_dataset():
    """Analyze the input data for all columns in dataset"""
    
    result = {}
    for i in range(9):
        result["column_{}".format(i)] = analyze_column(i)
    
    return result