from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, average_precision_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.compose import TransformedTargetRegressor
from sk_transformers import none_T,shrinkBigKTransformer,logminus_T,exp_T,logminplus1_T,logp1_T,dropConst,binaryYTransformer
from sk_missing_value_handler import missingValHandler

import logging
import numpy as np
from mylogger import myLogger



class sk_estimator:
    def __init__(self,):
        pass
    
    def linSvcClf(self,gridpoints=3):
        steps=[
            ('prep',missingValHandler()),
            ('scaler',StandardScaler()),
            ('shrink_k1',shrinkBigKTransformer()), # retain a subset of the best original variables
            ('polyfeat',PolynomialFeatures(interaction_only=0)), # create interactions among them
            ('drop_constant',dropConst()),
            ('reg',LinearSVC(random_state=0,tol=1e-4,max_iter=50000))]
        inner_pipeline=Pipeline(steps=steps)
        outer_pipeline=TransformedTargetRegressor(transformer=binaryYTransformer(),regressor=inner_pipeline,check_inverse=False)
        
        
        param_grid={
            'regressor__reg__C':np.logspace(-2,1,gridpoints),
            'regressor__shrink_k1__k_share':list(range(1,gridpoints+1)),
            'regressor__prep__strategy':['impute_middle','impute_knn_10']
        }
        inner_cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
        return GridSearchCV(outer_pipeline,param_grid=param_grid,cv=inner_cv)
    
    
    
    
    def rbfSvcClf(self,gridpoints=6):
        steps=[
            ('prep',missingValHandler()),
            ('scaler',StandardScaler()),
            ('reg',SVC(kernel='rbf',random_state=0,tol=1e-4,max_iter=50000, cache_size=2*10**5))]
        
        inner_pipeline=Pipeline(steps=steps)
        outer_pipeline=TransformedTargetRegressor(transformer=binaryYTransformer(),regressor=inner_pipeline,check_inverse=False)
        
        
        param_grid={
            'regressor__reg__C':np.logspace(-2,2,gridpoints), 
            'regressor__reg__gamma':np.logspace(-1,0.5,gridpoints),
            'regressor__prep__strategy':['impute_middle','impute_knn_10']
        }
        inner_cv=RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
        return GridSearchCV(outer_pipeline,param_grid=param_grid,cv=inner_cv)
    
    
    
    def gradientBoostingClf(self,):
        steps=[
            ('prep',missingValHandler(strategy='pass-through')),
            #('scaler',StandardScaler()),
            ('reg',GradientBoostingRegressor(random_state=0,n_estimators=10000,max_depth=4))]
        
        inner_pipeline=Pipeline(steps=steps)
        outer_pipeline=TransformedTargetRegressor(transformer=binaryYTransformer(),regressor=inner_pipeline,check_inverse=False)
        
        
        return outer_pipeline
    
    
    def linRegSupremeClf(self,gridpoints=3,binary_threshold=0):
        steps=[
            ('prep',missingValHandler()),
            ('scaler',StandardScaler()),
            ('shrink_k1',shrinkBigKTransformer()), # retain a subset of the best original variables
            ('polyfeat',PolynomialFeatures(interaction_only=0)), # create interactions among them

            ('drop_constant',dropConst()),
            ('shrink_k2',shrinkBigKTransformer(selector=ElasticNet())), # pick from all of those options
            ('reg',LinearSVR(random_state=0,tol=1e-4,max_iter=50000))]

        X_T_pipe=Pipeline(steps=steps)
        inner_cv=RepeatedKFold(n_splits=5, n_repeats=2, random_state=0)
        
        
        
        Y_T_X_T_pipe=TransformedTargetRegressor(transformer=binaryYTransformer(),regressor=X_T_pipe,check_inverse=False)
        Y_T__param_grid={
            'regressor__polyfeat__degree':[2],
            'regressor__shrink_k2__selector__alpha':np.logspace(-2,2,gridpoints),
            'regressor__shrink_k2__selector__l1_ratio':np.linspace(0,1,gridpoints),
            'regressor__shrink_k1__k_share':list(range(1,gridpoints+1)),
            'regressor__prep__strategy':['impute_middle','impute_knn_10']
        }
        

        
        lin_reg_Xy_transform=GridSearchCV(Y_T_X_T_pipe,param_grid=Y_T__param_grid,cv=inner_cv,n_jobs=-1)

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