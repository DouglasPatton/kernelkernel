import numpy as np


class k2dtool():
    """k2d refers to the fact that there will be kernels in kernels in these estimators"""
    def __init__(self,xdata,ydata,modeldict=none):
        self.x=xdata
        self.y=ydata
        self.n=np.shape(xdata)[0]
        self.p=np.shape(xdata)[1]
        self.result_tree=climb_model_tree()
        
    def climb_model_tree(self):
        "returns lists of lists of....dict of results"