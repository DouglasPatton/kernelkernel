from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.linear_model import Lasso,LogisticRegressionCV
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
from pi_db_tool import DBTool



class sk_estimator(myLogger):
    def __init__(self,scorer='f1_micro'):
        myLogger.__init__(self,name='sk_estimator.log')
        #self.logger.info('starting new sk_estimator log')
        #self.scorer_dict=SKToolInitializer().get_scorer_dict() # for gridsearchCV
        #self.est_dict=self.get_est_dict()
        
        self.scorer=scorer #from sk_tool_initializer obj
        
        self.name=None
        
    def get_spec_std(self,spec):
        db=DBTool().metadataDBdict()
        return db[spec]['X_train_std']
        
    def get_coef_from_fit_est(self,species,est_name,est,std_rescale=True):
        #std_rescale rescales coefficients from their cv sample std scale to the global
        ##xtrain(i.e., everything since so far running cv on all data) std scale.
        if std_rescale:
            global_std=self.get_spec_std(species)
        if est_name == 'linear-svc':
            coef=est.best_estimator_.regressor_['clf'].coef_.T
            if std_rescale:
                std=est.best_estimator_.regressor_['scaler'].scale_
                
        elif est_name == 'logistic-reg':
            coef=est['clf'].coef_.T
            if std_rescale:
                std=est['scaler'].scale_
                
        elif est_name=='linear-probability-model':
            coef=est.best_estimator_.regressor_['clf'].coef_.T
            if std_rescale:
                std=est.best_estimator_.regressor_['scaler'].scale_    
        else:assert False,f'unexpected est_name:{est_name}'  
        if std_rescale:
            self.logger.info(f'before transforming: coef.shape:{coef.shape}, std.shape:{std.shape}, global_std.shape:{global_std.shape}')
            #coef_raw=coef.copy()
            #std=std[:,None]
            #global_std=global_std.to_numpy()[:,None]
            #self.logger.info(f'coef:{coef}, std:{std}, global_std:{global_std}')
            global_std=global_std.to_numpy()
            coef=coef/std*global_std #rescaled
            self.logger.info(f'after scaling: coef.shape:{coef.shape}, std.shape:{std.shape}, global_std.shape:{global_std.shape}')
            coef[std<0.01]=0
            coef[global_std<0.01]=0
            #self.logger.info(f'est_name:{est_name}, std.T:{std}, coef_raw:{coef_raw}, coef:{coef}')
        
        return coef
    
        
    def get_est_dict(self,):
        fit_kwarg_dict={'clf__sample_weight':'balanced'}# all are the same now # move to pisces_params?
        estimator_dict={
            #add logistic with e-net
            'linear-probability-model':{'estimator':self.lpmClf,'fit_kwarg_dict':fit_kwarg_dict},#{'regressor__clf__sample_weight':'balanced'}},
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
    
        
    
    
    def linSvcClf(self,gridpoints=3,inner_cv_splits=5,inner_cv_reps=1,random_state=0):
        try:
            param_grid={
                'regressor__clf__C':np.logspace(-2,2,gridpoints),
            }
            inner_cv=RepeatedStratifiedKFold(n_splits=inner_cv_splits, n_repeats=inner_cv_reps, random_state=random_state)
            steps=[
                #
                ('scaler',StandardScaler()),
                #('shrink_k1',shrinkBigKTransformer(selector=LassorLarsCV(max_iter=128,cv=inner_cv))), # retain a subset of the best original variables
                #('polyfeat',PolynomialFeatures(interaction_only=0)), # create interactions among them
                #('drop_constant',dropConst()),
                ('clf',LinearSVC(random_state=random_state,tol=1e-2,max_iter=1000))]
            inner_pipeline=Pipeline(steps=steps)
            t_former=None#binaryYTransformer()
            outer_pipeline=TransformedTargetRegressor(transformer=t_former,regressor=inner_pipeline,check_inverse=False)
            static_pipeline=GridSearchCV(outer_pipeline,param_grid=param_grid,cv=inner_cv,scoring=self.scorer)
            return Pipeline(steps=[('prep',missingValHandler(strategy='impute_knn_10')),('static_pipeline',static_pipeline)])
        except:
            self.logger.exception('')
    
    
    
    
    def rbfSvcClf(self,gridpoints=3,inner_cv_splits=5,inner_cv_reps=1,random_state=0):
        try:
            steps=[
                ('scaler',StandardScaler()),
                ('clf',SVC(kernel='rbf',random_state=random_state,tol=1e-3,max_iter=2000, cache_size=2*10**4))]

            inner_pipeline=Pipeline(steps=steps)
            #t_former=None#binaryYTransformer()
            #outer_pipeline=TransformedTargetRegressor(transformer=t_former,regressor=inner_pipeline,check_inverse=False)


            param_grid={
                'clf__C':np.logspace(-2,2,gridpoints), 
                'clf__gamma':np.logspace(-2,0.5,gridpoints),
            }
            inner_cv=RepeatedStratifiedKFold(n_splits=inner_cv_splits, n_repeats=inner_cv_reps, random_state=random_state)
            static_pipeline= GridSearchCV(inner_pipeline,param_grid=param_grid,cv=inner_cv,scoring=self.scorer)
            return Pipeline(steps=[('prep',missingValHandler(strategy='impute_knn_10')),('static_pipeline',static_pipeline)])
        except:
            self.logger.exception('')
    
    def histGradientBoostingClf(self,gridpoints=3,inner_cv_splits=5,inner_cv_reps=1,random_state=0):
        try:
            steps=[('clf',HistGradientBoostingClassifier(random_state=random_state))]
            inner_pipeline=Pipeline(steps=steps)
            return inner_pipeline
        except:
            self.logger.exception('')
        
    def gradientBoostingClf(self,gridpoints=3,inner_cv_splits=5,inner_cv_reps=1,random_state=0):
        try:
            steps=[
                ('prep',missingValHandler(strategy='impute_knn_10')),
                #('scaler',StandardScaler()),
                ('clf',GradientBoostingClassifier(random_state=random_state,n_estimators=500,max_depth=4,ccp_alpha=0))]

            inner_pipeline=Pipeline(steps=steps)
            return inner_pipeline
        except:
            self.logger.exception('')
        
    def logisticClf(self,gridpoints=3,inner_cv_splits=5,inner_cv_reps=1,random_state=0):
        try:
            inner_cv=RepeatedStratifiedKFold(n_splits=inner_cv_splits, n_repeats=inner_cv_reps, random_state=random_state)
            steps=[
                ('prep',missingValHandler(strategy='impute_knn_10')),
                ('scaler',StandardScaler()),
                ('clf',LogisticRegressionCV(Cs=10,penalty='l1',solver='saga',max_iter=1000,cv=inner_cv))]

            
            return Pipeline(steps=steps)
        except:
            self.logger.exception('')
        
    def lpmClf(self,gridpoints=5,random_state=0,inner_cv_splits=5,inner_cv_reps=1):
        try:
            param_grid={'regressor__clf__alpha':np.logspace(np.log10(.01),np.log10(1000),2*gridpoints)}
            inner_cv=RepeatedStratifiedKFold(n_splits=inner_cv_splits, n_repeats=inner_cv_reps, random_state=random_state)
            steps=[
                #('prep',missingValHandler(strategy='impute_knn_10')),
                ('scaler',StandardScaler()),
                ('clf',Lasso(normalize=False,max_iter=1000,warm_start=True,random_state=random_state))]

            inner_pipeline=Pipeline(steps=steps)
            t_former=binaryYTransformer(threshold=0.5)
            t_pipe= TransformedTargetRegressor(transformer=t_former,regressor=inner_pipeline,check_inverse=False)
            static_pipeline= GridSearchCV(t_pipe,param_grid=param_grid,cv=inner_cv,scoring=self.scorer)
            return Pipeline(steps=[('prep',missingValHandler(strategy='impute_knn_10')),('static_pipeline',static_pipeline)])
            
        except:
            self.logger.exception('')
    
if __name__=="__main__":
    X, y= make_regression(n_samples=300,n_features=16,n_informative=10,noise=3)
    y=(y-np.mean(y))
    y01=binaryYTransformer(threshold=0.5).fit(y).transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y01, test_size=0.2, random_state=0)
    
    sk_est=sk_estimator()
    est_dict=sk_est.get_est_dict()
    print(est_dict)
    for est_name,est_setup_dict in est_dict.items():
        print(est_name)
        est=est_setup_dict['estimator']()
        est.fit(X_train,y_train)
        
        s=est.score(X_train,y_train)
        s_out=est.score(X_test,y_test)
        print('predictions',est.predict(X_test))
        print(f'for {est_name}')
        print(f'    fit r2 score: {s}')
        print(f'    test r2 score: {s_out}')
    
    #lrs=sk_estimator().rbf_svc_cv(gridpoints=3)                                                                          
    #lrs=sk_estimator().linear_svc_cv(gridpoints=3)                                     
    #lrs=sk_estimator().binLinRegSupreme(gridpoints=3)
    
    #cv=cross_validate(lrs,X,y01,scoring='r2',cv=2)
    #print(cv)
    #print(cv['test_score'])    