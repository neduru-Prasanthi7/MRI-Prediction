import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import sys
import os
import warnings
warnings.filterwarnings('ignore')
from log_code import setup_logging
logger = setup_logging('all_models')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve
import pickle

def knn(X_train,y_train,X_test,y_test):
    try:
      global knn_reg
      knn_reg = KNeighborsClassifier(n_neighbors=5)
      knn_reg.fit(X_train,y_train)
      logger.info(f'KNN Test Accuracy:{accuracy_score(y_test,knn_reg.predict(X_test))}')
      logger.info(f'KNN Classification report:{classification_report(y_test,knn_reg.predict(X_test))}')
      logger.info(f'KNN confusion matrix:{confusion_matrix(y_test,knn_reg.predict(X_test))}')
    except Exception as e:
      error_type, error_msg, error_line = sys.exc_info()
      logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def nb(X_train,y_train,X_test,y_test):
    try:
      global naive_reg
      naive_reg = GaussianNB()
      naive_reg.fit(X_train,y_train)
      logger.info(f'Naive Bayes Test Accuracy:{accuracy_score(y_test,naive_reg.predict(X_test))}')
      logger.info(f'Naive Bayes Classification report:{classification_report(y_test, naive_reg.predict(X_test))}')
      logger.info(f'Naive Bayesconfusion matrix:{confusion_matrix(y_test, naive_reg.predict(X_test))}')
    except Exception as e:
      error_type, error_msg, error_line = sys.exc_info()
      logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def lr(X_train,y_train,X_test,y_test):
    try:
      global lr_reg
      lr_reg = LogisticRegression()
      lr_reg.fit(X_train,y_train)
      logger.info(f'LogisticRegression Test Accuracy:{accuracy_score(y_test,lr_reg.predict(X_test))}')
      logger.info(f'LogisticRegression Classification report:{classification_report(y_test, lr_reg.predict(X_test))}')
      logger.info(f'LogisticRegression confusion matrix:{confusion_matrix(y_test, lr_reg.predict(X_test))}')

      with open('Mrri.pkl', 'wb') as f:
          pickle.dump(lr_reg,f)

    except Exception as e:
      error_type, error_msg, error_line = sys.exc_info()
      logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def dt(X_train,y_train,X_test,y_test):
    try:
      global dt_reg
      dt_reg = DecisionTreeClassifier(criterion='entropy')
      dt_reg.fit(X_train,y_train)
      logger.info(f'DecisionTreeClassifier Test Accuracy:{accuracy_score(y_test,dt_reg.predict(X_test))}')
      logger.info(f'DecisionTree Classification report:{classification_report(y_test, dt_reg.predict(X_test))}')
      logger.info(f'DecisionTree confusion matrix:{confusion_matrix(y_test, dt_reg.predict(X_test))}')
    except Exception as e:
      error_type, error_msg, error_line = sys.exc_info()
      logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def rf(X_train,y_train,X_test,y_test):
    try:
      global rf_reg
      rf_reg = RandomForestClassifier(n_estimators=5,criterion='entropy')
      rf_reg.fit(X_train,y_train)
      logger.info(f'RandomForestClassifier Test Accuracy:{accuracy_score(y_test,rf_reg.predict(X_test))}')
      logger.info(f'RandomForest Classification report:{classification_report(y_test, rf_reg.predict(X_test))}')
      logger.info(f'RandomForest confusion matrix:{confusion_matrix(y_test, rf_reg.predict(X_test))}')
    except Exception as e:
      error_type, error_msg, error_line = sys.exc_info()
      logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def ada(X_train,y_train,X_test,y_test):
    try:
      global ada_reg
      t = LogisticRegression()
      ada_reg = AdaBoostClassifier(estimator=t, n_estimators=5)
      ada_reg.fit(X_train, y_train)
      logger.info(f'AdaBoostClassifier Test Accuracy:{accuracy_score(y_test, ada_reg.predict(X_test))}')
      logger.info(f'ADaBoost Classification report:{classification_report(y_test, ada_reg.predict(X_test))}')
      logger.info(f'AdaBoost confusion matrix:{confusion_matrix(y_test, ada_reg.predict(X_test))}')

    except Exception as e:
      error_type, error_msg, error_line = sys.exc_info()
      logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def gb(X_train,y_train,X_test,y_test):
    try:
      global gb_reg
      gb_reg = GradientBoostingClassifier(n_estimators=5)
      gb_reg.fit(X_train,y_train)
      logger.info(f'GradientBoostingClassifier Test Accuracy:{accuracy_score(y_test,gb_reg.predict(X_test))}')
      logger.info(f'GradientBoosting Classification report:{classification_report(y_test, gb_reg.predict(X_test))}')
      logger.info(f'GradientBoosting confusion matrix:{confusion_matrix(y_test, gb_reg.predict(X_test))}')
    except Exception as e:
      error_type, error_msg, error_line = sys.exc_info()
      logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')
def xgb_(X_train,y_train,X_test,y_test):
    try:
      global xg_reg
      xg_reg = XGBClassifier()
      xg_reg.fit(X_train,y_train)
      logger.info(f'XGBClassifier Test Accuracy:{accuracy_score(y_test,xg_reg.predict(X_test))}')
      logger.info(f'XGB Classification report:{classification_report(y_test, xg_reg.predict(X_test))}')
      logger.info(f'XGB confusion matrix:{confusion_matrix(y_test, xg_reg.predict(X_test))}')
    except Exception as e:
      error_type, error_msg, error_line = sys.exc_info()
      logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

def common(X_train,y_train,X_test,y_test):
    try:
        knn(X_train, y_train, X_test, y_test)
        nb(X_train, y_train, X_test, y_test)
        lr(X_train, y_train, X_test, y_test)
        dt(X_train, y_train, X_test, y_test)
        rf(X_train, y_train, X_test, y_test)
        ada(X_train, y_train, X_test, y_test)
        gb(X_train, y_train, X_test, y_test)
        xgb_(X_train, y_train, X_test, y_test)

        # ---------------- ACCURACY ----------------
        acc_knn = accuracy_score(y_test, knn_reg.predict(X_test))
        acc_nb = accuracy_score(y_test, naive_reg.predict(X_test))
        acc_lr = accuracy_score(y_test, lr_reg.predict(X_test))
        acc_dt = accuracy_score(y_test, dt_reg.predict(X_test))
        acc_rf = accuracy_score(y_test, rf_reg.predict(X_test))
        acc_ada = accuracy_score(y_test, ada_reg.predict(X_test))
        acc_gb = accuracy_score(y_test, gb_reg.predict(X_test))
        acc_xgb = accuracy_score(y_test, xg_reg.predict(X_test))

        # ---------------- PROBABILITIES ----------------
        knn_prob = knn_reg.predict_proba(X_test)[:, 1]
        nb_prob = naive_reg.predict_proba(X_test)[:, 1]
        lr_prob = lr_reg.predict_proba(X_test)[:, 1]
        dt_prob = dt_reg.predict_proba(X_test)[:, 1]
        rf_prob = rf_reg.predict_proba(X_test)[:, 1]
        ada_prob = ada_reg.predict_proba(X_test)[:, 1]
        gb_prob = gb_reg.predict_proba(X_test)[:, 1]
        xgb_prob = xg_reg.predict_proba(X_test)[:, 1]

        # ---------------- ROC CURVES ----------------
        plt.figure(figsize=(6, 4))
        plt.plot([0, 1], [0, 1], 'k--')

        plt.plot(*roc_curve(y_test, knn_prob)[:2], label=f'KNN (Acc={acc_knn:.3f})')
        plt.plot(*roc_curve(y_test, nb_prob)[:2], label=f'NB (Acc={acc_nb:.3f})')
        plt.plot(*roc_curve(y_test, lr_prob)[:2], label=f'LR (Acc={acc_lr:.3f})')
        plt.plot(*roc_curve(y_test, dt_prob)[:2], label=f'DT (Acc={acc_dt:.3f})')
        plt.plot(*roc_curve(y_test, rf_prob)[:2], label=f'RF (Acc={acc_rf:.3f})')
        plt.plot(*roc_curve(y_test, ada_prob)[:2], label=f'ADA (Acc={acc_ada:.3f})')
        plt.plot(*roc_curve(y_test, gb_prob)[:2], label=f'GB (Acc={acc_gb:.3f})')
        plt.plot(*roc_curve(y_test, xgb_prob)[:2], label=f'XGB (Acc={acc_xgb:.3f})')

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve – Models with Accuracy")
        plt.legend(loc="lower right")
        plt.show()




    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')