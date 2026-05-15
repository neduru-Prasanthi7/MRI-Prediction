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

logger = setup_logging('balanced_data')
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def balanced_data(X_train, y_train, X_test, y_test):
    try:
        logger.info(f'Before class 1: {sum(y_train == 1)}')
        logger.info(f'Before class 0: {sum(y_train == 0)}')

        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

        # ✅ KEEP AS DATAFRAME
        X_train_bal = pd.DataFrame(X_train_bal, columns=X_train.columns)
        X_test = pd.DataFrame(X_test, columns=X_train.columns)

        y_train_bal = pd.Series(y_train_bal, name=y_train.name)
        y_test = pd.Series(y_test, name=y_test.name)

        logger.info(f'After X_train: {X_train_bal.shape}')
        logger.info(f'After y_train: {y_train_bal.shape}')

        logger.info(f'After class 1: {sum(y_train_bal == 1)}')
        logger.info(f'After class 0: {sum(y_train_bal == 0)}')

        return X_train_bal, y_train_bal, X_test, y_test

    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')








