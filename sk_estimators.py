from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.linear_model import ElasticNet, LinearRegression,LassoLarsCV,LogisticRegressionCV
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
#from sk_tool import SKToolInitializer
import logging
import numpy as np
from mylogger import myLogger



class sk_estimator(myLogger):
    def __init__(self,):
        myLogger.__init__(self,name='sk_estimator.log')
        #self.logger.info('starting new sk_estimator log')
        #self.scorer_dict=SKToolInitializer().get_scorer_dict() # for gridsearchCV
        #self.est_dict=self.get_est_dict()
        self.name=None
        
        
    def get_est_dict(self,):
        fit_kwarg_dict={'clf__sample_weight':'balanced'}# all are the same now
        estimator_dict={
            #add logistic with e-net
            'logistic-reg':{'estimator':self.logisticClf,'fit_kwarg_dict':fit_kwarg_dict},
            'linear-svc':{'estimator':self.linSvcClf,'fit_kwarg_dict':fit_kwarg_dict,},
            'rbf-svc':{'estimator':self.rbfSvcClf,'fit_kwarg_dict':fit_kwarg_dict,},
            'gradient-boosting-classifier':{'estimator':self.gradientBoostingClf,'fit_kwarg_dict':fit_kwarg_dict,},
            'hist-gradient-boosting-classifier':{'estimator':self.histGradientBoostingClf,'fit_kwarg_dict':fit_kwarg_dict,},
        }
        return estimator_dict
    
    """def get_coef_(self,est_name,est):
        coef_tool_dict={
            'logistic-reg':self.coef_linear
            'linear-svc':self.coef_linear
            'rbf-svc':
            'gradient-boosting-classifier':
            'hist-gradient-boosting-classifier':
        }
        return coef_tool_dict[name](est)
    
    def coef_linear(self,estimator):
        return estimator.coef_"""
    
        
    
    
    def linSvcClf(self,gridpoints=3,inner_cv_splits=10,inner_cv_reps=2,random_state=0):
        try:
            inner_cv=RepeatedStratifiedKFold(n_splits=inner_cv_splits, n_repeats=inner_cv_reps, random_state=random_state)
            steps=[
                ('prep',missingValHandler(strategy='impute_knn_10')),
                ('scaler',StandardScaler()),
                #('shrink_k1',shrinkBigKTransformer(selector=LassorLarsCV(max_iter=128,cv=inner_cv))), # retain a subset of the best original variables
                #('polyfeat',PolynomialFeatures(interaction_only=0)), # create interactions among them
                #('drop_constant',dropConst()),
                ('clf',LinearSVC(random_state=random_state,tol=1e-2,max_iter=1000))]
            inner_pipeline=Pipeline(steps=steps)
            t_former=None#binaryYTransformer()
            outer_pipeline=TransformedTargetRegressor(transformer=t_former,regressor=inner_pipeline,check_inverse=False)


            param_grid={
                'regressor__clf__C':np.logspace(-2,2,gridpoints),
                #'regressor__shrink_k1__k_share':[1,1/2,1/8],
                'regressor__prep__strategy':['impute_knn_10']
            }

            return GridSearchCV(outer_pipeline,param_grid=param_grid,cv=inner_cv,)
        except:
            self.logger.exception('')
    
    
    
    
    def rbfSvcClf(self,gridpoints=3,inner_cv_splits=10,inner_cv_reps=2,random_state=0):
        try:
            steps=[
                ('prep',missingValHandler(strategy='impute_knn_10')),
                ('scaler',StandardScaler()),
                ('clf',SVC(kernel='rbf',random_state=random_state,tol=1e-3,max_iter=2000, cache_size=2*10**4))]

            inner_pipeline=Pipeline(steps=steps)
            t_former=None#binaryYTransformer()
            outer_pipeline=TransformedTargetRegressor(transformer=t_former,regressor=inner_pipeline,check_inverse=False)


            param_grid={
                'regressor__clf__C':np.logspace(-2,2,gridpoints), 
                'regressor__clf__gamma':np.logspace(-2,0.5,gridpoints),
                'regressor__prep__strategy':['impute_knn_10']
            }
            inner_cv=RepeatedStratifiedKFold(n_splits=inner_cv_splits, n_repeats=inner_cv_reps, random_state=random_state)
            return GridSearchCV(outer_pipeline,param_grid=param_grid,cv=inner_cv,)
        except:
            self.logger.exception('')
    
    def histGradientBoostingClf(self,gridpoints=3,inner_cv_splits=10,inner_cv_reps=2,random_state=0):
        try:
            steps=[('clf',HistGradientBoostingClassifier(random_state=random_state))]
            inner_pipeline=Pipeline(steps=steps)
            return inner_pipeline
        except:
            self.logger.exception('')
        
    def gradientBoostingClf(self,gridpoints=3,inner_cv_splits=10,inner_cv_reps=2,random_state=0):
        try:
            steps=[
                ('prep',missingValHandler(strategy='impute_knn_10')),
                #('scaler',StandardScaler()),
                ('clf',GradientBoostingClassifier(random_state=random_state,n_estimators=500,max_depth=4,ccp_alpha=0))]

            inner_pipeline=Pipeline(steps=steps)
            return inner_pipeline
        except:
            self.logger.exception('')
        
    def logisticClf(self,gridpoints=3,inner_cv_splits=10,inner_cv_reps=2,random_state=0):
        try:
            steps=[
                ('prep',missingValHandler(strategy='impute_knn_10')),
                ('scaler',StandardScaler()),
                #('shrink_k1',shrinkBigKTransformer()), # retain a subset of the best original variables
                #('polyfeat',PolynomialFeatures(interaction_only=0)), # create interactions among them

                #('drop_constant',dropConst()),
                #('shrink_k2',shrinkBigKTransformer(selector=ElasticNet())), # pick from all of those options
                ('clf',LogisticRegressionCV(penalty='l1',solver='saga'))]

            '''X_T_pipe=Pipeline(steps=steps)
            inner_cv=RepeatedStratifiedKFold(n_splits=inner_cv_splits, n_repeats=inner_cv_reps, random_state=random_state)



            Y_T_X_T_pipe=TransformedTargetRegressor(transformer=binaryYTransformer(),regressor=X_T_pipe,check_inverse=False)
            Y_T__param_grid={
                #'regressor__polyfeat__degree':[2],
                #'regressor__shrink_k2__selector__alpha':np.logspace(-2,2,gridpoints),
                #'regressor__shrink_k2__selector__l1_ratio':np.linspace(0,1,gridpoints),
                #'regressor__shrink_k1__k_share':[1,1/2,1/8],
                #'regressor__prep__strategy':['impute_knn_10']
            }



            lin_reg_Xy_transform=GridSearchCV(Y_T_X_T_pipe,param_grid=Y_T__param_grid,cv=inner_cv,)'''

            return Pipeline(steps=steps)
        except:
            self.loggger.exception('')
        

    
if __name__=="__main__":
    X, y= make_regression(n_samples=3000,n_features=16,n_informative=3,noise=3)
    y=(y-np.mean(y))
    y01=binaryYTransformer().fit(y).transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y01, test_size=0.2, random_state=0)
    
    sk_est=sk_estimator()
    est_dict=sk_est.get_est_dict()
    for est_name,est_setup_dict in est_dict.items():
        est=est_setup_dict['estimator']()
        est.fit(X_train,y_train)
        s=est.score(X_train,y_train)
        s_out=est.score(X_test,y_test)
        print(f'for {est_name}')
        print(f'    fit r2 score: {s}')
        print(f'    test r2 score: {s_out}')
    
    #lrs=sk_estimator().rbf_svc_cv(gridpoints=3)                                                                          
    #lrs=sk_estimator().linear_svc_cv(gridpoints=3)                                     
    #lrs=sk_estimator().binLinRegSupreme(gridpoints=3)
    
    #cv=cross_validate(lrs,X,y01,scoring='r2',cv=2)
    #print(cv)
    #print(cv['test_score'])    