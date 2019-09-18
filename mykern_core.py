import mykern as mk

data1=generate 1+ 3dimensional joint random distributions //
so we can smooth it and compare to real dist or use 2 dimensions to predict the 3rd.


#create modeldict for 1 layer of ddiffs,
#product of kernels of each parameter (as in liu and yang eq1.)

modeldict1={'max_ddiff':2, 'kern_grid':'no', 'ddiff_kern':gaussian,//
            'Kh_form':'exp_l2'  'hyper_param_form_dict':{'ddiff_exp':'fixed_list', 'x_bandwidth':'full'}]
            }
#exp_l2 means take the l2 ("el two") norm across all parameters and then plug into the kernel (diff from liu and yang)
#x_params:full means each plus constant;
    #each means 1 hyper parameter per column of data
#kern_grid:'n' makes n evenly spaced values across every parameter for density estimation.
    #'no' means use original data, which is useful for calibrating hyper parameters
hyper_paramdict1={'ddiff_exp_params':np.array([-1,1]),'x_bandwidth_params':np.ones(data1.shape[1]),//
                  }

#create hyper parameter optimization
#with 1 bandwidth-hyperparameter per
#regression parameter as in liu and yang eq1
optimize_dict1={'method':'Nelder-Mead','hyper_paramdict':hyper_paramdict1,'modeldict':modeldict1}


hyper_params_out=mk.kNdtool.optimize_hyper_params(mk.MY_KDEreg,data,optimize_dict1)#this needs to store hyperparameters 



