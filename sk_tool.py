from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, average_precision_score
from sklearn.ensemble import GradientBoostingRegressor
from sk_transformers import none_T,shrinkBigKTransformer,logminus_T,exp_T,logminplus1_T,logp1_T,dropConst
from sk_missing_value_handler import missingValHandler
from sklearn.base import BaseEstimator, TransformerMixin
#from sk_estimators import linRegSupremeClf,linSvcClf, rbfSvcClf, gradientBoostingClf
from sk_estimators import sk_estimator as sk_est
import logging
import numpy as np
from mylogger import myLogger
import re
from warnings import filterwarnings
filterwarnings('ignore')

class SKToolInitializer(myLogger):
    def __init__(self,model_gen):
        myLogger.__init__(self,name='skToolInitializer.log')
        self.logger.info('starting skToolInitializer logger')
        self.model_gen=model_gen
        self.scorer_list=self.get_scorer_list()
        
    def get_scorer_list(self):
        return ['f1_micro','precision_micro','recall_micro','accuracy']
        
    def run(self,datagen_obj):
        sktool=SkTool(self.model_gen)
        if datagen_obj.cv:
            self.logger.info(f'starting cv for sktool.model_gen["name"]{sktool.model_gen["name"]}')
            return cross_validate(sktool,datagen_obj.X_train,datagen_obj.y_train,cv=datagen_obj.cv,return_estimator=True,scoring=self.scorer_list)
        else:
            self.logger.info(f'starting simple fit for sktool.model_gen["name"]{sktool.model_gen["name"]}')
            return sktool.fit(datagen_obj.X_train,datagen_obj.y_train)

    
class SkTool(BaseEstimator,TransformerMixin,myLogger,):
    def __init__(self,model_gen=None):
        myLogger.__init__(self,name='SkTool.log')
        self.logger.info('starting SkTool logger')
        self.model_gen=model_gen
        self.name=model_gen['name']
        
    def transform(self,X,y=None):
        return self.model_.transform(X,y)
    def score(self,X,y,sample_weight=None):
        return self.model_.score(X,y,sample_weight) # don't need sample_weights unless scoring other than f1_micro or similar used
    def predict(self,X):
        return self.model_.predict(X)
     
   
    def fit(self,X,y):
        try:
            modelgen=self.model_gen
            est_name=modelgen['name']
            self.model_kwargs=modelgen['kwargs']
            sk_estimator=sk_est()
            est_dict=sk_estimator.get_est_dict()
            self.est=est_dict[est_name]['estimator']
            fit_kwarg_dict=est_dict[est_name]['fit_kwarg_dict']
            self.fit_kwargs_=self.make_fit_kwargs(fit_kwarg_dict,X,y)
            self.n_,self.k_=X.shape
            #self.logger(f'self.k_:{self.k_}')
            self.model_=self.est(**self.model_kwargs)
            self.logger.info(f'starting fit of {est_name}')
            self.model_.fit(X,y,**self.fit_kwargs_)
            self.logger.info(f'completed fit of {est_name}')
            return self
        except:
            self.logger.exception('fit error')
            self.logger.error(f'X.shape:{X.shape}, X:{X}')
    
    
    
    def make_fit_kwargs(self,fit_kwarg_dict,X,y):
        keys=list(fit_kwarg_dict.keys())
        for key in keys:
            if re.search('sample_weight',key):
                if fit_kwarg_dict[key] =='balanced':
                    fit_kwarg_dict[key]=self.make_sample_weight(y)
                elif fit_kwarg_dict[key] is None:
                    fit_kwarg_dict.pop(key)
                else:
                    assert False,f'expecting sample_weights to be None or "balanced" but sample_weight:{fit_kwarg_dict[key]} at key:{key}'
        return fit_kwarg_dict
    
    def make_sample_weight(self,y):
        wt=np.empty(y.shape,dtype=np.float64)
        cats=np.unique(y)
        c=cats.size
        n=y.size
        for u in cats:
            share=n/(np.size(y[y==u])*c)
            wt[y==u]=share
        return wt
            
        
        
        
            
    
            
