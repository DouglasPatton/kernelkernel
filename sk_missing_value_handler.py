import logging
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression, Lars
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer,KNNImputer
from sk_transformers import none_T,shrinkBigKTransformer,logminus_T,exp_T,logminplus1_T,logp1_T,dropConst



class missingValHandler(BaseEstimator,TransformerMixin):
    # https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#use-columntransformer-by-selecting-column-by-names
    def __init__(self,strategy='impute_middle',transformer=None):
        self.strategy=strategy
        self.transformer=transformer
        self.logger=logging.getLogger()
    def fit(self,X,y):
        if type(X)!=pd.DataFrame:
            X=pd.DataFrame(X)
        self.X_dtypes_=dict(X.dtypes)
        
        #self.obj_idx=[i for i,(var,dtype) in enumerate(self.X_dtypes_.items()) if dtype=='object']
        
        #separate variables by continuous,categorical(unordered)
        self.obj_idx_=[i for i,(var,dtype) in enumerate(self.X_dtypes_.items()) if dtype=='object']
        self.float_idx_=[i for i in range(X.shape[1]) if i not in self.obj_idx_]
        self.cat_list=[X.iloc[:,idx].unique() for idx in self.obj_idx_]
        x_nan_count=X.isnull().sum().sum() # sums by column and then across columns
        try:
            y_nan_count=y.isnull().sum().sum()
        except:
            try:y_nan_count=np.isnan(y).sum()
            except:
                if not y is None:
                    y_nan_count='error'
                    self.logger.exception(f'error summing nulls for y, type(y):{type(y)}')
                else:
                    pass
                
        cat_encoder=OneHotEncoder(categories=self.cat_list,sparse=False,) # drop='first'
        xvars=list(X.columns)
        if type(self.strategy) is str:
            if self.strategy=='drop':
                assert False, 'develop drop columns with >X% missing vals then drop rows with missing vals'
                
            if self.strategy=='pass-through':
                numeric_T=('no_transform',none_T(),self.float_idx_)
                categorical_T=('cat_drop_hot',cat_encoder,self.obj_idx_)
            if self.strategy=='drop_row':
                X=X.dropna(axis=0) # overwrite it
                
                numeric_T=('no_transform',none_T(),self.float_idx_)
                categorical_T=('cat_drop_hot',cat_encoder,self.obj_idx_)
                
            if self.strategy=='impute_middle':
                numeric_T=('num_imputer', SimpleImputer(strategy='mean'),self.float_idx_)
                cat_imputer=make_pipeline(SimpleImputer(strategy='most_frequent'),cat_encoder)
                categorical_T=('cat_imputer',cat_imputer,self.obj_idx_)
            if self.strategy[:10]=='impute_knn':
                if len(self.strategy)==10:
                    k=5
                else:
                    k=int(''.join([char for char in self.strategy[10:] if char.isdigit()])) #extract k from the end
                numeric_T=('num_imputer', KNNImputer(n_neighbors=k),self.float_idx_)
                cat_imputer=make_pipeline(SimpleImputer(strategy='most_frequent'),cat_encoder)
                categorical_T=('cat_imputer',cat_imputer,self.obj_idx_)
        self.T_=ColumnTransformer(transformers=[numeric_T,categorical_T])
        self.T_.fit(X,y)        
            
                
            
        #self.logger.info(f'x_nan_count:{x_nan_count}, y_nan_count:{y_nan_count}')
        return self
        
    def transform(self,X,y=None):
        if type(X)!=pd.DataFrame:
            X=pd.DataFrame(X)
        
        
        X=self.T_.transform(X)
        x_nan_count=np.isnan(X).sum() # sums by column and then across columns
        """try:
            y_nan_count=y.isnull().sum().sum()
        except:
            y_nan_count='error'
            self.logger.exception('error summing nulls for y')"""
        if x_nan_count>0:
            self.logger.info(f'x_nan_count is non-zero! x_nan_count:{x_nan_count}')
        #print(X)
        return X