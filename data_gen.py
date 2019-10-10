
import numpy as np

class data_gen():
    '''generates numpy arrays of random training or validation for model: y=xb+e or variants
    '''
    #def __init__(self, data_shape=(200,5), ftype='linear', xval_size='same', sparsity=0, xvar=1, xmean=0, evar=1, betamax=10):
    def __init__(self, data_shape=None, ftype='linear'):
        if data_shape==None:
            self.n=200;self.p=2
        else:self.n=data_shape[0];self.p=data_shape[1]
        p=self.p
        n=self.n
        betamax=100
        evar=1.5
        xtall=3
        xwide=2
        spreadx=np.random.randint(xwide, size=(1,p))+1#random row vector to multiply by each random column of x to allow s.d. upto 5
        shiftx=np.random.randint(0,xtall, size=(1,p))-xtall/2#random row vector to add to each column of x
        randx=np.random.randn(n,p)
        self.x = np.concatenate((np.ones((n,1)),shiftx+spreadx*randx),axis=1)
        #generate error~N(0,1)
        self.e=np.random.randn(n)*evar**.5
        

        #make beta integer, non-zero
        self.b=(np.random.randint(betamax, size=(p+1,))+1)*(2*np.random.randint(2, size=(p+1,))-np.ones(p+1,)) #if beta is a random integer, it could be 0
        #make a simple y for testing
        self.y=np.matmul(self.x, self.b)+self.e