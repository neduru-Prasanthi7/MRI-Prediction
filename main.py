import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('main')

from random_sample import complete_random_sample_technique
from var_out import variable_transformation_outliers
from balance_data import balanced_data
from all_models import common
from sklearn.preprocessing import StandardScaler
import pickle

class MRI:
    def __init__(self, path):
        try:
           self.path = path
           self.df = pd.read_csv(self.path)

           logger.info(f'Data Loaded Successfully')
           logger.info(f'Total number of rows :{self.df.shape[0]}')
           logger.info(f'Total number of columns :{self.df.shape[1]}')
           logger.info(f'{self.df.info()}')
           logger.info(f'checking null values :{self.df.isnull().sum()}')

           #for i in self.df.columns:
            #   logger.info(f'{self.df[i].dtype}')

           #independent
           self.X = self.df.iloc[:,:-1]
           self.y = self.df.iloc[:,-1]

           self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size = 0.2,random_state=42)

           self.y_train = pd.Series(self.y_train).round().astype(int)
           self.y_test = pd.Series(self.y_test).round().astype(int)

           logger.info(f'X_train shape :{self.X_train.shape}')
           logger.info(f'X_test shape :{self.X_test.shape}')
           logger.info(f'X_train columns :{self.X_train.columns}')
           logger.info(f'X_test columns :{self.X_test.columns}')
           logger.info(f'y_train shape :{self.y_train.shape}')
           logger.info(f'y_test shape :{self.y_test.shape}')

           logger.info(f'{self.y_train.sample(5)}')
           logger.info(f'{self.y_test.sample(5)}')


        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def missing_values(self):
        try:
            if self.X_train.isnull().any().any() or self.X_test.isnull().any().any()>0:
                self.X_train, self.X_test = complete_random_sample_technique(
                    self.X_train, self.X_test
                )
            else:
                logger.info('No missing values')
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(
                f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}'
            )
    def vt_out(self):
        try:
            #for i in self.X_train.columns:
             #   logger.info(f'{self.X_train[i].dtype}')
            logger.info(f'X_train columns :{self.X_train.columns}')
            logger.info(f'X_test columns :{self.X_test.columns}')
            self.X_train,self.X_test = variable_transformation_outliers(self.X_train,self.X_test)
            logger.info(f'X_train_num columns :{self.X_train.columns}')
            logger.info(f'X_test_num columns :{self.X_test.columns}')
            logger.info(f"After vt_out: {type(self.X_train)}")
        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    def data_balance(self):
        try:
            logger.info(f'X_train columns :{self.X_train.columns}')
            logger.info(f'{self.y_train.unique()}')

            self.X_train,self.y_train,self.X_test,self.y_test = balanced_data(self.X_train,self.y_train,self.X_test,self.y_test)

            logger.info(f"Balanced X_train shape: {self.X_train.shape}")
            logger.info(f"Balanced y_train shape: {self.y_train.shape}")
            logger.info(f"After data_balance: {type(self.X_train)}")
            logger.info(f'null values:{self.X_train.isnull().sum()}')

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

    from sklearn.preprocessing import StandardScaler

    def scaling(self):
        try:
            # Ensure DataFrame
            if isinstance(self.X_train, np.ndarray):
                self.X_train = pd.DataFrame(self.X_train, columns=self.X.columns)

            if isinstance(self.X_test, np.ndarray):
                self.X_test = pd.DataFrame(self.X_test, columns=self.X.columns)

            logger.info(f'data scaling')

            scale_cols = self.X_train.select_dtypes(
                include=['int64', 'float64']
            ).columns

            logger.info(f'scaling columns : {scale_cols.tolist()}')

            sc = StandardScaler()
            self.X_train[scale_cols] = sc.fit_transform(self.X_train[scale_cols])
            self.X_test[scale_cols] = sc.transform(self.X_test[scale_cols])

            logger.info(f'X_train shape :{self.X_train.shape}')
            logger.info(f'X_test shape :{self.X_test.shape}')

            with open('scalar.pkl', 'wb') as f:
                pickle.dump(sc, f)

            common(self.X_train,self.y_train,self.X_test, self.y_test)

        except Exception as e:
            error_type, error_msg, error_line = sys.exc_info()
            logger.info( f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')


if __name__ == '__main__':
    try:
        obj = MRI(f'C:\\Users\\nedur\\Downloads\\MRI Prediction\\Mri_dataset.csv')
        obj.missing_values()
        obj.vt_out()
        obj.data_balance()
        obj.scaling()


    except Exception as e:
        error_type, error_msg, error_line = sys.exc_info()
        logger.info(f'Error in Line no : {error_line.tb_lineno}: due to {error_msg}')

