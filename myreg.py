import numpy as np

class reg():
    

        def myreg(self,y,x):
            # a simple regression that prints its own mse and returns estimated parameters
            q=np.linalg.inv(np.matmul(x.T,x))
            self.b=np.matmul(np.matmul(q,x.T),y)
            self.mse=np.sum((y-x@self.b)**2)/y.shape[0]

        def myregstdz(self,y,x): 
            [n,k]=np.shape(x)
            onecol=0
            xfull=x
            if np.allclose(np.ones([n,1]),x[:,0]):#fix problems with standardizing column of 1's
                x=x[:,1:]
                onecol=1;
            xmean=np.mean(x,axis=0)
            xvar=np.var(x,axis=0,ddof=1)#subtracting xmean isn't necessary, as it happens anyways
            onevec=np.ones([n,1])
            yvar=np.var(y, ddof=1)
            sy=np.power(yvar,.5)
            sx=np.power(xvar,.5)
            ymean=np.mean(y)        
            xstdz=(1/sx)*(x-xmean)
            ystdz=(1/sy)*(y-ymean)
            Q=np.linalg.inv(np.matmul(xstdz.T,xstdz))
            bhat1=np.matmul(np.matmul(Q,xstdz.T),ystdz)
            bhat=sy*bhat1*(1/sx)#undo beta coef. by multiplying each by Sy/Sx
            if onecol==1:
                bhat_int=ymean-np.sum(sy*xmean*bhat1*(1/sx))
                bhat=np.concatenate([[bhat_int],bhat])
            self.mse=np.sum((y-xfull@bhat)**2)/y.shape[0]
            self.b=bhat

            def myregpredict(y,x,yval,xval):
    # a simple regression that prints its own mse and returns estimated parameters
    q=np.linalg.inv(np.matmul(x.T,x))
    bhat=np.matmul(np.matmul(q,x.T),y)
    mse=np.sum((yval-xval@bhat)**2)/yval.shape[0]
    #print('mse=',mse)
    return mse
