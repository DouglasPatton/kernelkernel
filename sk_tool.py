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

    
class SkTool(BaseEstimator,TransformerMixin,myLogger,):
    def __init__(self,model_gen=None):
        myLogger.__init__(self,name='skTool.log')
        self.logger.info('starting skTool logger')
        if not model_gen is None:
            self.model_gen=model_gen
        
        
    def fit(self,X,y):
        modelgen=self.model_gen
        est_name=modelgen['name']
        self.model_kwargs=modelgen['kwargs']
        est_dict=self.get_est_dict()
        self.est=est_dict[est_name]['estimator']
        fit_kwarg_dict=est_dict[est_name]['fit_kwarg_dict']
        self.fit_kwargs=self.make_fit_kwargs(fit_kwarg_dict,X,y)
        self.n_,self.k_=X.shape
        #self.logger(f'self.k_:{self.k_}')
        self.model_=self.est(**self.model_kwargs)
        self.model_.fit(X,y,**self.fit_kwargs)
        return self
    def transform(self,X,y=None):
        return self.model_.transform(X,y)
    def score(self,X,y):
        return self.model_.score(X,y,) # don't need sample_weights unless scoring other than f1_micro or similar used
    def predict(self,X):
        return self.model_.predict(X)
    
    def get_est_dict(self,):
        estimator_dict={
            #add logistic with e-net
            fit_kwarg_dict={'regressor__clf__sample_weight':'balanced'}# all are the same now
            'lin-reg-classifier':{'estimator':linRegSupremeClf,'fit_kwarg_dict':fit_kwarg_dict},
            'linear-svc':{'estimator':linSvcClf,'fit_kwarg_dict':fit_kwarg_dict,
            'rbf-svc':{'estimator':rbfSvcClf,'fit_kwarg_dict':fit_kwarg_dict,
            'gradient-boosting-classifier':{'estimator':gradientBoostingClf,'fit_kwarg_dict':fit_kwarg_dict,
            'hist-gradient-boosting-classifier':{'estimator':histGradientBoostingClf,'fit_kwarg_dict':fit_kwarg_dict,
        }
    
    def make_fit_kwargs(self,fit_kwarg_dict,X,y):
        keys=list(fit_kwarg_dict.keys())
        for key in keys:
            if re.search('sample_weight',key):
                if fit_kwarg_dict[key] =='balanced':
                    fit_kwarg_dict[key]=self.make_sample_weight(y)
                else:
                    assert False,f'expecting sample_weights to be "balanced" but sample_weight:{fit_kwarg_dict[key]} at key:{key}'
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
            
        
        
        
            
    
            
