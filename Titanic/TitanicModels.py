# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:16:44 2019

@author: rkshatri
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import time
import xgboost as xgb
from xgboost import XGBClassifier
import pandas_ml as pdml
from TitanicDataTransformer import TitanicDataTransformer


def read_data(filepath,test=False) :

    titanic_df = pd.read_csv(filepath,dtype={'Name':str,'Age':np.float64,'SibSp':np.float64,'Pclass':np.float64,'Parch':np.float64,'Fare':np.float64,'Sex':str,'Embarked':str,'Cabin':str})
    if not test:
        return titanic_df.drop(columns='Survived'),titanic_df['Survived']
    else :
        return titanic_df

def get_missing_cols(train,test) :
    missing_cols = set(train.columns) - set(test.columns)
    
    for c in missing_cols :
        test[c] = 0
    
    return test[train.columns]
   

def train_logistic_reg(X_train,y_train,X_test,y_test,X_eval,eval_pids,out_to_file) :
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train,y_train)

    test_pred = log_reg.predict(X_test)
    eval_pred = log_reg.predict(X_eval)
    
    class_report = classification_report(y_true=y_test,y_pred=test_pred)
    accuracy = accuracy_score(y_true=y_test,y_pred=test_pred)
    conf_matrix = confusion_matrix(y_true=y_test,y_pred=test_pred)
    
    print("Logistic regression classification report:\n",class_report)
    print("Accuracy: ",accuracy)
    print("Confusion matrix:\n",conf_matrix)
    
    if out_to_file :
        test_output = pd.concat([eval_pids,pd.DataFrame(eval_pred)],axis=1)
        test_output.to_csv('test_output/log_reg/log_reg_'+str(datetime.now().day)+str(time.time())+'.csv',index=False,header=['PassengerId','Survived'])

def train_decision_tree(X_train,y_train,X_test,y_test,X_eval,eval_pids,out_to_file) :
    
    parameters = {'criterion':('gini','entropy'),'max_depth':[ 3, 4, 5, 6, 8, 10, 12, 15],'min_samples_split':range(2,21),'min_samples_leaf':range(1,6),'presort':(True,False)}
    dec_tree = tree.DecisionTreeClassifier()
    kfolds = StratifiedKFold(7)
    clf = GridSearchCV(dec_tree,parameters,cv=kfolds.split(X_train,y_train),scoring='f1')
    clf.fit(X_train,y_train)
    
    test_pred = clf.predict(X_test)
    eval_pred = clf.predict(X_eval)
    
    class_report = classification_report(y_true=y_test,y_pred=test_pred)
    accuracy = accuracy_score(y_true=y_test,y_pred=test_pred)
    conf_matrix = confusion_matrix(y_true=y_test,y_pred=test_pred)
    
    print("Decision tree classification report:\n",class_report)
    print("Accuracy: ",accuracy)
    print("Confusion matrix:\n",conf_matrix)
    
    if out_to_file :
        test_output = pd.concat([eval_pids,pd.DataFrame(eval_pred)],axis=1)
        test_output.to_csv('test_output/dec_tree/dec_tree_'+str(datetime.now().day)+str(time.time())+'.csv',index=False,header=['PassengerId','Survived'])

def train_random_forest(X_train,y_train,X_test,y_test,X_eval,eval_pids,out_to_file) :
    
    parameters = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
    rand_forest = RandomForestClassifier()
    #rand_forest.fit(X_train,y_train)
    kfolds = StratifiedKFold(7)
    clf = RandomizedSearchCV(rand_forest,parameters,cv=kfolds.split(X_train,y_train))
    clf.fit(X_train,y_train)
   
    
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    eval_pred = clf.predict(X_eval)
    
    class_report = classification_report(y_true=y_test,y_pred=test_pred)
    accuracy_train = accuracy_score(y_true=y_train,y_pred=train_pred)
    accuracy = accuracy_score(y_true=y_test,y_pred=test_pred)
    conf_matrix = confusion_matrix(y_true=y_test,y_pred=test_pred)
    
    print("Random Forest classification report:\n",class_report)
    print("Accuracy: ",accuracy)
    print(accuracy_train)
    print("Confusion matrix:\n",conf_matrix)
    
    if out_to_file :
        test_output = pd.concat([eval_pids,pd.DataFrame(eval_pred)],axis=1)
        test_output.to_csv('test_output/rand_forest/rand_forest_'+str(datetime.now().day)+str(time.time())+'.csv',index=False,header=['PassengerId','Survived'])

def train_xg_boost(X_train,y_train,X_test,y_test,X_eval,eval_pids,out_to_file) :

    parameters = {'learning_rate': [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,'max_depth': [ 3, 4, 5, 6, 8, 10, 12, 15],'min_child_weight' : [ 1, 3, 5, 7 ],'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ] }
    train = xgb.DMatrix(X_train,label=y_train)
    test = xgb.DMatrix(X_test,label=y_test)
    evalu = xgb.DMatrix(X_eval)
    
    evallist = [(test,'eval'),(train,'train')]
    #bst = xgb.train(param,train,40,evallist,early_stopping_rounds=10)
    xgbclass = XGBClassifier(n_estimators=600,objective='binary:logistic',metrics=('mae'),silent=False)
    bst = RandomizedSearchCV(xgbclass,parameters,cv=7).fit(X_train,y_train)
    #xgbclass.fit(train,y_train)

    test_pred = bst.predict(X_test)
    test_pred_labels = pd.Series([pred>=0.5 for pred in test_pred],index=y_test.index).astype(int)
    eval_pred = bst.predict(X_eval)
    eval_pred_labels = pd.Series([pred>=0.5 for pred in eval_pred]).astype(int)
    
    class_report = classification_report(y_true=y_test,y_pred=test_pred_labels)
    accuracy = accuracy_score(y_true=y_test,y_pred=test_pred_labels)
    conf_matrix = confusion_matrix(y_true=y_test,y_pred=test_pred_labels)
    
    print("XGBoost classification report:\n",class_report)
    print("Accuracy: ",accuracy)
    print("Confusion matrix:\n",conf_matrix)
    
    if out_to_file :
        test_output = pd.concat([eval_pids,pd.DataFrame(eval_pred_labels)],axis=1)
        test_output.to_csv('test_output/xgboost/xgboost_'+str(datetime.now().day)+str(time.time())+'.csv',index=False,header=['PassengerId','Survived'])
    return

def main(argv=None) :
    my_dir = '.'
    train_path = '/data/train.csv'
    test_path = '/data/test.csv'


    X,y = read_data(my_dir+train_path)
    X_eval = read_data(my_dir+test_path,test=True)
    DT = TitanicDataTransformer()
    X = DT.fit_transform(X)

    X_eval = DT.fit_transform(X_eval)
    eval_pids = X_eval['PassengerId']
    X.drop(columns='PassengerId',inplace=True)
    X_eval.drop(columns='PassengerId',inplace=True)
  
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25)
    
    #One Hot Encode
    cat_vars = ['Sex','Embarked','Title','Deck']
    for feature in cat_vars :
        X_train = pd.concat([X_train,pd.get_dummies(X_train[feature],prefix=feature)],axis=1)
        X_train.drop(feature,1,inplace=True)
        X_eval = pd.concat([X_eval,pd.get_dummies(X_eval[feature],prefix=feature)],axis=1)
        X_eval.drop(feature,1,inplace=True)
        X_test = pd.concat([X_test,pd.get_dummies(X_test[feature],prefix=feature)],axis=1)
        X_test.drop(feature,1,inplace=True)
    
    X_test = get_missing_cols(X_train,X_test)
    X_eval = get_missing_cols(X_train,X_eval)

    #feature scaling
    scaler = StandardScaler()
    X_train[['Pclass','Age','SibSp','Parch','Fare','Family_Size','Age*Class','Fare_per_person']]= scaler.fit_transform(X_train[['Pclass','Age','SibSp','Parch','Fare','Family_Size','Age*Class','Fare_per_person']])
    X_test[['Pclass','Age','SibSp','Parch','Fare','Family_Size','Age*Class','Fare_per_person']] = scaler.transform(X_test[['Pclass','Age','SibSp','Parch','Fare','Family_Size','Age*Class','Fare_per_person']])
    X_eval[['Pclass','Age','SibSp','Parch','Fare','Family_Size','Age*Class','Fare_per_person']] = scaler.transform(X_eval[['Pclass','Age','SibSp','Parch','Fare','Family_Size','Age*Class','Fare_per_person']])
    
    #Logistic Regression
    train_logistic_reg(X_train,y_train,X_test,y_test,X_eval,eval_pids,False)

    #Decision Tree
    train_decision_tree(X_train,y_train,X_test,y_test,X_eval,eval_pids,False)
    
    #Random Forest
    train_random_forest(X_train,y_train,X_test,y_test,X_eval,eval_pids,False)
    
    #XG Boost
    train_xg_boost(X_train,y_train,X_test,y_test,X_eval,eval_pids,True)

    
if __name__ == '__main__':
    main()