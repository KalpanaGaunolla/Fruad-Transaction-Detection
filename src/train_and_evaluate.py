# load the train and test
# train algo
# save the metrices, params
from math import gamma
from operator import mod
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest,RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from get_data import read_params
import argparse
import joblib
import json


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    clf_report = classification_report(actual, pred)
    roc_auc = roc_auc_score(actual, pred)
    return accuracy,clf_report,roc_auc

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")


    train_y = train[target]
    train_y=train_y.iloc[0:50001,:]
    fraud=train_y[train_y['Class']==1]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    train_x=train_x.iloc[0:50001,:]
    test_x = test.drop(target, axis=1)
    state=np.random.RandomState(42)
    X_outliers=state.uniform(low=0,high=1,size=(train_x.shape[0],train_x.shape[1]))
    
    print(train_x.shape)

    ##Build Isolation Forest Model
    #ifmod=IsolationForest(n_estimators=100,random_state=1,verbose=0,contamination=0.02)
    #if_model=ifmod.fit(train_x)
    
    #Build Random Forest Model
    rf_model=RandomForestClassifier(n_estimators=100,min_samples_split=5,random_state=1).fit(train_x,train_y)

    predicted_val=rf_model.predict(test_x)
    outliers_pred=rf_model.predict(X_outliers)

    (rf_accuracy, rf_clf_report, rf_roc_auc) = eval_metrics(test_y, predicted_val)

    print("rf_accuracy:",rf_accuracy)
    print("rf_clf_report:",rf_clf_report)
    print("rf_roc_auc:", rf_roc_auc)
    print("no_outliers:",len(fraud))

    ## Local Outlier Factor (LOF)
    #lof_model=LocalOutlierFactor(n_neighbors=20,algorithm='auto',leaf_size=30,p=2,metric_params=None,contamination=0.0017)

    #lof_predict=lof_model.fit_predict(train_x,train_y)
    
    #(lof_accuracy, lof_clf_report, lof_roc_auc) = eval_metrics(test_y, lof_predict)

    #print("lof_accuracy:", lof_accuracy)
    #print("lof_clf_report:",lof_clf_report)
    #print("lof_roc_auc:", lof_roc_auc)

    ## Support Vector Machine (SVM)
    #svm_model=OneClassSVM(kernel='rbf',degree=3,gamma=0.1,nu=0.05,max_iter=-1).fit(train_x,train_y)

    #svm_predict=svm_model.predict(test_x)
    #outliers_pred_svm=svm_model.predict(X_outliers)

    #(svm_accuracy, svm_clf_report, svm_roc_auc) = eval_metrics(test_y, svm_predict)

    #print("svm_accuracy:",svm_accuracy)
    #print("svm_clf_report:",svm_clf_report)
    #print("svm_roc_auc:", svm_roc_auc)
    #print("No_outliers:",len(fraud))

#####################################################################3

    scores_file = config["reports"]["scores"]

    with open(scores_file, "w") as f:
        scores = {
            "RF accuracy": rf_accuracy,
            "RF clf_report": rf_clf_report,
            "RF roc_auc": rf_roc_auc,
            "RF outliers":len(fraud)
            #"lof_accuracy":lof_accuracy,
            #"lof clf report":lof_clf_report,
            #"lof roc acu":lof_roc_auc,
            #"svm accuracy":svm_accuracy,
            #"svm clf report":svm_clf_report,
            #"svm roc auc":svm_roc_auc,
            #"svm outliers":len(outliers_pred_svm)
        }
        json.dump(scores, f, indent=4)

    os.makedirs(model_dir, exist_ok=True)
    rf_model_path = os.path.join(model_dir, "rf_model.joblib")
    #lof_model_path = os.path.join(model_dir, "lof_model.joblib")
    #svm_model_path=os.path.join(model_dir,"svm_model.joblib")

    joblib.dump(rf_model, rf_model_path)
    #joblib.dump(lof_model,lof_model_path)
    #joblib.dump(svm_model,svm_model_path)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
