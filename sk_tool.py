from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, average_precision_score
from sklearn.ensemble import GradientBoostingRegressor
from sk_transformers import none_T,shrinkBigKTransformer,logminus_T,exp_T,logminplus1_T,logp1_T,dropConst
from sk_missing_value_handler import missingValHandler
from sk_estimators import linRegSupremeClf,linSvcClf, rbfSvcClf, gradientBoostingClf
import logging
import numpy as np
from mylogger import myLogger

    
class skTool(BaseEstimator,TransformerMixin,myLogger,):
    def __init__(self,modeldict):
        myLogger.__init__(self,name='skTool.log')
        self.logger.info('starting skTool logger')
        self.modeldict=modeldict
        
        
    def fit(self,X,y):
        modelgen=self.modeldict['modelgen']
        est_name=modelgen['name']
        self.kwargs=modelgen['kwargs']
        est_dict=self.get_est_dict()
        self.est=est_dict[est_name]
        self.n_,self.k_=X.shape
        #self.logger(f'self.k_:{self.k_}')
        self.model_=self.est(**self.kwargs)
        self.model_.fit(X,y)
        return self
    def transform(self,X,y=None):
        return self.model_.transform(X,y)
    def score(self,X,y):
        return self.model_.score(X,y)
    def predict(self,X):
        return self.model_.predict(X)
    
    def get_est_dict(self,):
        estimator_dict={
            'lin-reg-classifier':linRegSupremeClf,
            'linear-svc':linSvcClf, 
            'rbf-svc':rbfSvcClf, 
            'gradient-boosting-classifier':gradientBoostingClf
        }
    
        
        
            
    
            
d