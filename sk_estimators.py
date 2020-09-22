from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import cross_validate, train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, average_precision_score
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingClassifier,HistGradientBoostingClassifier
from sklearn.datasets import make_regression
from sklearn.compose import TransformedTargetRegressor
from sk_transformers import none_T,shrinkBigKTransformer,logminus_T,exp_T,logminplus1_T,logp1_T,dropConst,binaryYTransformer
from sk_missing_value_handler import missingValHandler

import logging
import numpy as np
from mylogger import myLogger



class sk_estimator(myLogger):
    def __init__(self,):
        myLogger.__init__(self,name='sk_estimator.log')
        self.logger.info('starting new sk_estimator log')
        #self.est_dict=self.get_est_dict()
        
        
    def get_est_dict(self,):
        fit_kwarg_dict={'regressor__clf__sample_weight':'balanced'}# all are the same now
        estimator_dict={
            #add logistic with e-net
            'lin-reg-classifier':{'estimator':self.linRegSupremeClf,'fit_kwarg_dict':fit_kwarg_dict},
            'linear-svc':{'estimator':self.linSvcClf,'fit_kwarg_dict':fit_kwarg_dict,},
            'rbf-svc':{'estimator':self.rbfSvcClf,'fit_kwarg_dict':fit_kwarg_dict,},
            'gradient-boosting-classifier':{'estimator':self.gradientBoostingClf,'fit_kwarg_dict':fit_kwarg_dict,},
            'hist-gradient-boosting-classifier':{'estimator':self.histGradientBoostingClf,'fit_kwarg_dict':fit_kwarg_dict,},
        }
        return estimator_dict
    
    def linSvcClf(self,gridpoints=3,inner_cv_splits=10,inner_cv_reps=2,):
        steps=[
            ('prep',missingValHandler()),
            ('scaler',StandardScaler()),
            ('shrink_k1',shrinkBigKTransformer()), # retain a subset of the best original variables
            ('polyfeat',PolynomialFeatures(interaction_only=0)), # create interactions among them
            ('drop_constant',dropConst()),
            ('clf',LinearSVC(random_state=0,tol=1e-4,max_iter=50000))]
        inner_pipeline=Pipeline(steps=steps)
        outer_pipeline=TransformedTargetRegressor(transformer=binaryYTransformer(),regressor=inner_pipeline,check_inverse=False)
        
        
        param_grid={
            'regressor__clf__C':np.logspace(-2,1,gridpoints),
            'regressor__shrink_k1__k_share':[1,1/2,1/8],
            'regressor__prep__strategy':['impute_knn_5']
        }
        inner_cv=RepeatedStratifiedKFold(n_splits=inner_cv_splits, n_repeats=inner_cv_reps, random_state=0)
        return GridSearchCV(outer_pipeline,param_grid=param_grid,cv=inner_cv,scoring='f1_micro')
    
    
    
    
    def rbfSvcClf(self,gridpoints=3,inner_cv_splits=10,inner_cv_reps=2,):
        steps=[
            ('prep',missingValHandler()),
            ('scaler',StandardScaler()),
            ('clf',SVC(kernel='rbf',random_state=0,tol=1e-4,max_iter=50000, cache_size=2*10**5))]
        
        inner_pipeline=Pipeline(steps=steps)
        outer_pipeline=TransformedTargetRegressor(transformer=binaryYTransformer(),regressor=inner_pipeline,check_inverse=False)
        
        
        param_grid={
            'regressor__clf__C':np.logspace(-2,2,gridpoints), 
            'regressor__clf__gamma':np.logspace(-1,0.5,gridpoints),
            'regressor__prep__strategy':['impute_knn_5']
        }
        inner_cv=RepeatedStratifiedKFold(n_splits=inner_cv_splits, n_repeats=inner_cv_reps, random_state=0)
        return GridSearchCV(outer_pipeline,param_grid=param_grid,cv=inner_cv,scoring='f1_micro')
    
    def histGradientBoostingClf(self,gridpoints=3,inner_cv_splits=10,inner_cv_reps=2,):
        steps=[('clf',HistGradientBoostingClassifier())]
        inner_pipeline=Pipeline(steps=steps)
        outer_pipeline=TransformedTargetRegressor(transformer=binaryYTransformer(),regressor=inner_pipeline,check_inverse=False)
        
        param_grid={'regressor__clf__min_samples_leaf':[10,15,20],
                   'regressor__l2_regularization':np.logspace(-2,2,gridpoints)}
        inner_cv=RepeatedStratifiedKFold(n_splits=inner_cv_splits, n_repeats=inner_cv_reps, random_state=0)
        return GridSearchCV(outer_pipeline,param_grid=param_grid,cv=inner_cv,scoring='f1_micro')
        
    def gradientBoostingClf(self,gridpoints=3,inner_cv_splits=10,inner_cv_reps=2,):
        steps=[
            ('prep',missingValHandler()),
            #('scaler',StandardScaler()),
            ('clf',GradientBoostingClassifier(random_state=0,n_estimators=10000,max_depth=4,ccp_alpha=0))]
        
        inner_pipeline=Pipeline(steps=steps)
        outer_pipeline=TransformedTargetRegressor(transformer=binaryYTransformer(),regressor=inner_pipeline,check_inverse=False)
        
        param_grid={
            'regressor__clf__C':np.logspace(-2,2,gridpoints), 
            'regressor__clf__ccp_alpha':np.logspace(-3,-1,gridpoints),
        }
        inner_cv=RepeatedStratifiedKFold(n_splits=inner_cv_splits, n_repeats=inner_cv_reps, random_state=0)
        return GridSearchCV(outer_pipeline,param_grid=param_grid,cv=inner_cv,scoring='f1_micro')
        
    
    
    def linRegSupremeClf(self,gridpoints=3,inner_cv_splits=10,inner_cv_reps=2,):
        steps=[
            ('prep',missingValHandler()),
            ('scaler',StandardScaler()),
            ('shrink_k1',shrinkBigKTransformer()), # retain a subset of the best original variables
            ('polyfeat',PolynomialFeatures(interaction_only=0)), # create interactions among them

            ('drop_constant',dropConst()),
            ('shrink_k2',shrinkBigKTransformer(selector=ElasticNet())), # pick from all of those options
            ('clf',LinearRegression())]

        X_T_pipe=Pipeline(steps=steps)
        inner_cv=RepeatedStratifiedKFold(n_splits=inner_cv_splits, n_repeats=inner_cv_reps, random_state=0)
        
        
        
        Y_T_X_T_pipe=TransformedTargetRegressor(transformer=binaryYTransformer(),regressor=X_T_pipe,check_inverse=False)
        Y_T__param_grid={
            'regressor__polyfeat__degree':[2],
            'regressor__shrink_k2__selector__alpha':np.logspace(-2,2,gridpoints),
            'regressor__shrink_k2__selector__l1_ratio':np.linspace(0,1,gridpoints),
            'regressor__shrink_k1__k_share':[1,1/2,1/8],
            'regressor__prep__strategy':['impute_knn_5']
        }
        

        
        lin_reg_Xy_transform=GridSearchCV(Y_T_X_T_pipe,param_grid=Y_T__param_grid,cv=inner_cv,n_jobs=-1,scoring='f1_micro')

        return lin_reg_Xy_transform
    
    
    
if __name__=="__main__":
    X, y= make_regression(n_samples=300,n_features=6,n_informative=3,noise=1)
    y=(y-np.mean(y))
    y01=binaryYTransformer().fit(y).transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y01, test_size=0.2, random_state=0)
    
    lrs=sk_estimator().gradient_boosting_classifier()                                                                          
    #lrs=sk_estimator().rbf_svc_cv(gridpoints=3)                                                                          
    #lrs=sk_estimator().linear_svc_cv(gridpoints=3)                                     
    #lrs=sk_estimator().binLinRegSupreme(gridpoints=3)
    lrs.fit(X_train,y_train)
    s=lrs.score(X_train,y_train)
    
    s_out=lrs.score(X_test,y_test)
    print(f'fit r2 score: {s}')
    print(f'test r2 score: {s_out}')
    #cv=cross_validate(lrs,X,y01,scoring='r2',cv=2)
    #print(cv)
    #print(cv['test_score'])    