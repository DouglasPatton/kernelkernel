#this document is really structured more as a jupyter notebook, but there is no markdown yet.

import mykern as mk

data1=generate mixed 3dimensional joint random distributions //
so we can smooth it and compare to real dist or use 2 dimensions to predict the 3rd.


#create modeldict for 1 layer of Ndiffs,
#product of kernels of each parameter (as in liu and yang eq1.)
n,p=data1.shape
modeldict1={
    'max_bw_Ndiff':2,
    'normalize_Ndiffwtsum':'own_n',
    'kern_grid':'no',
    'outer_kern':'gaussian',
    'Ndiff_bw_kern':'rbfkern',
    'regression_model':'NW'
    'hyper_param_form_dict':{
        'Ndiff_exponent':'fixed',
        'p_bandwidth':'free',
        'all_x_bandwidth':'free'
        'all_y_bandwidth':'free'
        }
    }
    
#max_bw_Ndiff: is the depth of Ndiffs applied in estimating the bandwidth.
#'normalize_Ndiffwtsum':
    #'across' means sum across own level of kernelized-Ndiffs and divide by that sum (CDF approach)
    #'own_n' means for (n+k)diff where n+K=max_bw_Ndiff
#kern_grid:
    #no means smooth the original data
    #yes means form a grid across all the variables for xin, yxin
#Ndiff_bw_kern:
    #rbfkern means use the radial basis function kernel
    #'product' means use product kernel like as in liu and yang eq1. 
#
#'regression_model':
    #'NW' means use nadaraya-watson kernel regression
    #'full_logit' means local logit with all variables entering linearly
    #'rbf_logit' means local logit with 1 parameters: scaled l2 norm centered on zero (globally or by i?). Is this a new idea?
#kern_grid
    #if int, then create int evnely spaced values from -3 to 3 (standard normal middle ~99%)
    #'no' means use original data, which is useful for calibrating hyper parameters
#hyper_param_form_dict is a nested dictionary



#------------------------------------------
#-----starting hyper parameter values------
hyper_paramdict1={
    'Ndiff_exponent':np.array([-1,1]),
    'p_bandwidth':np.ones([p,]),
    'all_x_bandwidth':0.3,
    'all_y_bandwidth'=0.3)
                  }

#create hyper parameter optimization
#with 1 bandwidth-hyperparameter per
#regression parameter 
optimizedict1={'method':'Nelder-Mead','hyper_param_dict':hyper_paramdict1,'model_dict':modeldict1}

#-----------------Calibrate/Optimize--------
#-------find values for hyperparameters-----
optimized_Ndiff_kernel=kNdtool.optimize_hyper_params(ydata,xdata,optimizedict1)

'''
#plot and compare
on synthetic data, 1, 2, 3+ mixed distributions
1d - Ndiff vs gaussian kernel vs kernel_tunneling
2d -  


multidimensional x problem
e.g., parameter treatment
"product kernel approach" vs l2 "el two" (radial basis?)distance 
'''
