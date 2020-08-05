from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.svm import LinearSVR, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, train_test_split, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, average_precision_score
from sklearn.ensemble import GradientBoostingRegressor
import logging

class skTool:
    def __init__(self):
        pass
        
    def predictY(self,xtrain,xtest,ytrain,ytest):
        xtrain=xtrain[:,:-1]# drop the spatial variable
        xtest=xtest[:,:-1]
        try:
            self.logger=logging.getLogger(__name__)
            self.logger.info('started sk_tool object')
        except:
            logdir=os.path.join(os.getcwd(),'log')
            if not os.path.exists(logdir): os.mkdir(logdir)
            handlername=os.path.join(logdir,f'sk_tool-{self.pname}.log')
            logging.basicConfig(
                handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=100)],
                level=logging.DEBUG,
                format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                datefmt='%Y-%m-%dT%H:%M:%S')

            #handler=logging.RotatingFileHandler(os.path.join(logdir,handlername),maxBytes=8000, backupCount=5)
            self.logger = logging.getLogger(handlername)
        try:
            linear_regression=make_pipeline(StandardScaler(),LinearRegression())
            elastic_net = make_pipeline(StandardScaler(), ElasticNetCV())
            
            #linear_svr = make_pipeline(StandardScaler(),GridSearchCV(LinearSVR(random_state=0,tol=1e-3,max_iter=5000),n_jobs=1,param_grid={'C':np.linspace(-1,4,8)}))
            linear_svr = Pipeline(steps=[('scaler',StandardScaler()),('lin_svr',LinearSVR(random_state=0,tol=1e-4,max_iter=10000))])
            lin_svr_param_grid={'lin_svr__C':np.logspace(-2,2,5)}  
            linear_svr_CV= GridSearchCV(linear_svr,param_grid=lin_svr_param_grid)

            rbf_svr=Pipeline(steps=[('scaler',StandardScaler()),('rbf_svr',SVR(kernel='rbf',tol=1e-4,max_iter=10000, cache_size=2*10**3))])
            rbf_svr_param_grid={'rbf_svr__C':np.logspace(-2,2,5),
                               'rbf_svr__gamma':np.logspace(-1,0.5,5)} 
            rbf_svr_CV=GridSearchCV(rbf_svr,param_grid=rbf_svr_param_grid)
                                  
            gradient_boosting_reg=make_pipeline(GradientBoostingRegressor())
            
            
            
            model_dict={'linear-regression':linear_regression,
                'elastic-net':elastic_net, 
                'linear-svr':linear_svr_CV, 
                'rbf-svr':rbf_svr_CV, 
                'gradient-boosting-reg':gradient_boosting_reg}
            loss_fn_dict={'mse':mean_squared_error, 'mae':mean_absolute_error, 'r2':r2_score, 'f1_score':f1_score, 'average_precision_score':average_precision_score}
            yhatdict={}
            for model_name,model in model_dict.items():
                model.fit(xtrain,ytrain)
                ytesthat=model.predict(xtest)
                yhatdict[model_name]=ytesthat
                #yhatdict[model_name]={lf_name:lf(ytest,ytesthat) for lf_name,lf in loss_fn_dict.items()}
            return yhatdict
        except:
            self.logger.exception('')
            
        
            