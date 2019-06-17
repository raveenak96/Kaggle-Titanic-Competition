# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 16:31:42 2019

@author: rkshatri
"""

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def substrings_in_string(big_string,substrings) :
    for substring in substrings :
        if big_string.find(substring) != -1 :
            return substring
    return np.nan

def replace_titles(data) :
    title = data['Title']
    if title in ['Major','Rev','Col','Capt','Don','Jonkheer'] :
        return 'Mr'
    elif title in ['Ms','Mlle'] :
        return 'Miss'
    elif title in ['Mme','Countess'] :
        return 'Mrs'
    elif title == 'Dr' :
        if data['Sex'] == 'male' :
            return 'Mr'
        else:
            return 'Mrs'
    else :
        return title
    
class TitanicDataTransformer(TransformerMixin,BaseEstimator) :
    
    def __init__(self,verbose=False) :
        
        self.verbose = verbose
    
    def fit(self,X,y=None) :
        if(self.verbose):
            print("Verbose mode on!")
        return self
    
    
    
    def transform(self,X,y=None) :
        X.drop(columns='Ticket',inplace=True)
    
        #fill na values
        X.Age.fillna(X.Age.mean(),inplace=True)
        X.Fare.fillna(X.Fare.mean(),inplace=True)
        X.Embarked.fillna('Unknown',inplace=True)
    
    
        #feature engineering
        titles = ['Mr','Mrs','Miss','Master','Major', 'Rev',
                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                    'Don', 'Jonkheer']
        X['Title'] = X['Name'].map(lambda x: substrings_in_string(x,titles))
        X['Title'] = X.apply(replace_titles,axis = 1)
    
        X.Cabin.fillna('Unknown',inplace=True)
        cabins = ['A','B','C','D','E','F','T','G','Unknown']
        X['Deck'] = X['Cabin'].map(lambda x: substrings_in_string(x,cabins))
        X['Family_Size'] = X['SibSp'] + X['Parch']
        X['Age*Class'] = X['Age']*X['Pclass']
    
        X['Fare_per_person'] = X['Fare'] / (X['Family_Size'] + 1)
        
        X['IsAlone'] = (X['Family_Size']==0).astype(int)

        return X.drop(columns=['Name','Cabin'])
    
    