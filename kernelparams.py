import numpy as np
from pisces_data_huc12 import PiscesDataTool

class KernelParams:
    
    def __init__(self,):
        self.n=32 #used to generate variations datagen-batch_n and ykern_grid that are len n and n+1
        self.batchcount_variation_list=[8]
    def getoptdictvariations(self,source='monte'):
        max_bw_Ndiff=2
        
               
        ''''hyper_param_form_dict':{
                'Ndiff_exponent':'free',
                'x_bandscale':'non-neg',
                'Ndiff_depth_bw':'non-neg',
                'outer_x_bw':'non-neg',
                'outer_y_bw':'non-neg',
                'y_bandscale':'fixed'
                '''
        
        #hyper_param_form_dict_variations=('modeldict:hyper_param_form_dict:x_bandscale',['fixed'])
        #Ndiff_exponentstartingvalue_variations=('hyper_param_dict:Ndiff_exponent',[np.array([0,0]),.5*np.array([-1,1]),1.8*np.array([-1,1])])
        Ndiff_exponentstartingvalue_variations=('hyper_param_dict:Ndiff_exponent',[factor*np.array([-1,1]) for factor in [.2,.5,.9,1.3]])
        
        #Ndiff_exponentstartingvalue_variations=('hyper_param_dict:Ndiff_exponent',[1.8*np.array([-1,1])])
        Ndiff_outer_x_bw_startingvalue_variations=('hyper_param_dict:outer_x_bw',[np.array([.7]),np.array([.2])])
        #Ndiff_outer_x_bw_startingvalue_variations=('hyper_param_dict:outer_x_bw',[np.array([.5])])
        Ndiff_outer_y_bw_startingvalue_variations=('hyper_param_dict:outer_y_bw',[np.array([.7]),np.array([.3])])
        #Ndiff_outer_y_bw_startingvalue_variations=('hyper_param_dict:outer_y_bw',[np.array([.5])])
                                  
        #NWnorm_variations=('modeldict:NWnorm',['across','none'])
        NWnorm_variations=('modeldict:NWnorm',['across-except:batchnorm'])
        #binary_y_variations=('modeldict:binary_y',[0.5])
        binary_y_variations=('modeldict:binary_y',[(.5, .25, .125, 'avgavg', 'avgmedian')]) # if binary_y is a tuple,
        #   then optimization chooses continuous phat and calculates alternative MSEs. 'avgavg' means
        #   calculate the avg phat for 0 and for 1 and avg those for the threshold.
        loss_function_variations=('modeldict:loss_function',['batchnorm_crossval'])
        #loss_function_variations=('modeldict:loss_function',['batch_crossval'])
        #cross_mse,cross_mse2
        #loss_function_variations=('modeldict:loss_function',['batch_crossval'])
        Ndiff_type_variations = ('modeldict:Ndiff_type', ['product','recursive'])
        #Ndiff_type_variations = ('modeldict:Ndiff_type', ['recursive'])
        max_bw_Ndiff_variations = ('modeldict:max_bw_Ndiff', [max_bw_Ndiff])
        Ndiff_start_variations = ('modeldict:Ndiff_start', [1])
        product_kern_norm_variations = ('modeldict:product_kern_norm', ['none','self'])
        #product_kern_norm_variations = ('modeldict:product_kern_norm', ['none','own_n'])
        normalize_Ndiffwtsum_variations = ('modeldict:normalize_Ndiffwtsum', ['none','own_n','across'])
        #normalize_Ndiffwtsum_variations = ('modeldict:normalize_Ndiffwtsum', ['none'])
        
        if source=='monte':
            standardization_variations=('modeldict:std_data',['all'])
            ykerngrid_form_variations=('modeldict:ykerngrid_form',[('even',4),('exp',4)])
            ykern_grid_variations=('modeldict:ykern_grid',[self.n+1,'no'])
            regression_model_variations=('modeldict:regression_model',['NW','NW-rbf2','NW-rbf'])
            
            
            optdict_variation_list = [
                                      Ndiff_outer_x_bw_startingvalue_variations,
                                      Ndiff_outer_y_bw_startingvalue_variations,
                                      ykerngrid_form_variations,
                                      NWnorm_variations,
                                      loss_function_variations,
                                      regression_model_variations, 
                                      product_kern_norm_variations,
                                      normalize_Ndiffwtsum_variations,
                                      Ndiff_type_variations,
                                      ykern_grid_variations,
                                      max_bw_Ndiff_variations,
                                      Ndiff_start_variations,
                                      standardization_variations,
                                     ]#hyper_param_form_dict_variations,
                                     
        if source=='pisces':
            #standardization_variations=('modeldict:std_data',[([],'float')])#a tuple containing lists of variables to standardize in y,x. 'float' means standardize all variables that are floats rather than string
            #standardization_variations=('modeldict:std_data',[([],'float'),([0],'float')])#[i] means standardize the ith variable. for y it can only be [0] or [] for no std
            standardization_variations=('modeldict:std_data',[([],'float')])#[i] means standardize the ith variable. for y it can only be [0] or [] for no std
            ykerngrid_form_variations=('modeldict:ykerngrid_form',[('binary',)])
            ykern_grid_variations=('modeldict:ykern_grid',[2])
            regression_model_variations=('modeldict:regression_model',['NW'])#add logistic when developed fully
            #regression_model_variations=('modeldict:regression_model',['NW'])#add logistic when developed fully
            spatialtransform_variations=('modeldict:spatialtransform',[('ln1',)])#
        
            optdict_variation_list = [binary_y_variations,
                                      Ndiff_exponentstartingvalue_variations,
                                      Ndiff_outer_x_bw_startingvalue_variations,
                                      Ndiff_outer_y_bw_startingvalue_variations,
                                      ykerngrid_form_variations,
                                      NWnorm_variations,
                                      loss_function_variations,
                                      regression_model_variations, 
                                      product_kern_norm_variations,
                                      normalize_Ndiffwtsum_variations,
                                      Ndiff_type_variations,
                                      ykern_grid_variations,
                                      max_bw_Ndiff_variations,
                                      Ndiff_start_variations,
                                      standardization_variations,
                                      spatialtransform_variations
                                     ]
            #hyper_param_form_dict_variations,
        return optdict_variation_list

    def getdatagenvariations(self,source='monte'):
        if source=='monte':
            #the default datagen_dict as of 11/25/2019
            #datagen_dict={'batch_n':32,'batchcount':10, 'param_count':param_count,'seed':1, 'ftype':'linear', 'evar':1, 'source':'monte'}
            batch_n_variations=('batch_n',[self.n])
            batchcount_variations=('batchcount',[4])
            ftype_variations=('ftype',['linear','quadratic'])
            param_count_variations=('param_count',[2,4])
            datagen_variation_list=[batch_n_variations,batchcount_variations,ftype_variations,param_count_variations]
        if source=='pisces':
            try:self.specieslist
            except:
                pdh12=PiscesDataTool()
                self.specieslist=pdh12.returnspecieslist()
            
                
                
                
            #species_variations=('species',self.specieslist)
            species_variations=('species',[self.specieslist[i] for i in range(0,len(self.specieslist),4)])
            # print('species_variations',species_variations)
            
            batch_n_variations=('batch_n',[self.n])
            batchcount_variations=('batchcount',self.batchcount_variation_list)
            datagen_variation_list=[batch_n_variations,batchcount_variations,species_variations]
        return datagen_variation_list
    
    def build_hyper_param_start_values(self,modeldict):
        max_bw_Ndiff=modeldict['max_bw_Ndiff']
        Ndiff_start=modeldict['Ndiff_start']
        Ndiff_param_count=max_bw_Ndiff-(Ndiff_start-1)
        p=modeldict['param_count']
        assert not p==None, f"p is unexpectedly p:{p}"
        if modeldict['Ndiff_type']=='product':
                hyper_paramdict1={
                'Ndiff_exponent':-.1*np.ones([Ndiff_param_count,]),
                'x_bandscale':1*np.ones([p,]),
                'outer_x_bw':np.array([0.3,]),
                'outer_y_bw':np.array([0.3,]),
                'Ndiff_depth_bw':.3*np.ones([Ndiff_param_count,]),
                'y_bandscale':1.0*np.ones([1,])
                    }

        if modeldict['Ndiff_type']=='recursive':
            hyper_paramdict1={
                'Ndiff_exponent':0.001*np.ones([Ndiff_param_count,]),
                'x_bandscale':1*np.ones([p,]),
                'outer_x_bw':np.array([0.3,]),
                'outer_y_bw':np.array([0.3,]),
                'Ndiff_depth_bw':np.array([0.3]),
                'y_bandscale':1.0*np.ones([1,])
                }
        return hyper_paramdict1
            
                               
    def setdata(self, source):#creates the initial datagen_dict
        
        if source is None or source=='monte':
            source='monte'
            datagen_dict={
                'source':'monte',
                'validate_batchcount':10,
                'batch_n':self.n,
                'batchcount':10,  
                'param_count':2,
                'seed':1, 
                'ftype':'linear', 
                'evar':1                
                                }
        elif source=='pisces':
            #floatselecttup=(2,3,5,6)
            #floatselecttup=()
            floatselecttup=(3,5,6)
            spatialselecttup=(8,)
            param_count=len(floatselecttup)+len(spatialselecttup)
            datagen_dict={
                'source':'pisces',
                'batch_n':self.n,
                'batchcount':8, #for batch_crossval and batchnorm_crossval, this specifies the number of groups of batch_n observations to be used for cross-validation. 
                'sample_replace':'no', #if no, batches are created until all data is sampled, and sampling with replacement used to fill up the last batch
                #if 'no-drop' then drop any observations that don't fit into a batch (not developed)
                'species':'all',
                #'species':'all', #could be 'all', int for the idx or a string with the species name. if 'all', then variations of datagen_dict will be created from pdh12.specieslist
                'missing':'drop_row', #drop the row(observation) if any data is missing
                'floatselecttup':floatselecttup,
                'spatialselecttup':spatialselecttup,
                'param_count':param_count
            }
        else: 
            assert False, f"error, source not recognized. source:{source}"
            
        self.datagen_dict=datagen_dict
        return datagen_dict    


    def build_optdict(self,opt_dict_override=None,param_count=None,species=None):
        if opt_dict_override==None:
            opt_dict_override={}
        max_bw_Ndiff=2
        Ndiff_start=1
        Ndiff_param_count=max_bw_Ndiff-(Ndiff_start-1)
        modeldict1={
            'binary_y':None, # if not None, then specifies the threshold of p(y=1|x) for predicting 1, e.g., 0.5
            'std_data':'all',
            'loss_function':'mse',
            'Ndiff_type':'product',
            'param_count':param_count,
            'Ndiff_start':Ndiff_start,
            'max_bw_Ndiff':max_bw_Ndiff,
            'normalize_Ndiffwtsum':'own_n',
            'NWnorm':'across',
            'xkern_grid':'no',
            'ykern_grid':33,
            'maxbatchbatchcount':1,
            'outer_kern':'gaussian',
            'Ndiff_bw_kern':'rbfkern',
            'outer_x_bw_form':'one_for_all',
            'regression_model':'NW',
            'product_kern_norm':'self',
            'hyper_param_form_dict':{
                'Ndiff_exponent':'free',
                'x_bandscale':'non-neg',
                'Ndiff_depth_bw':'non-neg',
                'outer_x_bw':'non-neg',
                'outer_y_bw':'non-neg',
                'y_bandscale':'fixed'
                }
            }
        
        """modeldict1={
            'binary_y':None, # if not None, then specifies the threshold of p(y=1|x) for predicting 1, e.g., 0.5
            'std_data':'all',
            'loss_function':'mse',
            'Ndiff_type':'product',
            'param_count':param_count,
            'Ndiff_start':Ndiff_start,
            'max_bw_Ndiff':max_bw_Ndiff,
            'normalize_Ndiffwtsum':'own_n',
            'NWnorm':'across',
            'xkern_grid':'no',
            'ykern_grid':33,
            'maxbatchbatchcount':1,
            'outer_kern':'gaussian',
            'Ndiff_bw_kern':'rbfkern',
            'outer_x_bw_form':'one_for_all',
            'regression_model':'NW',
            'product_kern_norm':'self',
            'hyper_param_form_dict':{
                'Ndiff_exponent':'free',
                'x_bandscale':'non-neg',
                'Ndiff_depth_bw':'non-neg',
                'outer_x_bw':'non-neg',
                'outer_y_bw':'non-neg',
                'y_bandscale':'fixed'
                }
            }"""
        if not species is None:
            modeldict1['species']=species
            modeldict1['spatialtransform']=('divide',1)
        #hyper_paramdict1=self.build_hyper_param_start_values(modeldict1)
        hyper_paramdict1={}
        
        #optimization settings for Nelder-Mead optimization algorithm
        optiondict_NM={
            'xatol':0.05,
            'fatol':.01,
            'adaptive':True,
            'maxiter':200
            }
        optimizer_settings_dict1={
            'method':'Nelder-Mead',
            'options':optiondict_NM,
            'mse_threshold':'naive_mse',
            'help_start':0,
            'partial_match':0,
            'do_minimize':0 # do_minimize=0 means just predict once for mse and don't optimize
            }
        
        optimizedict1={
            'opt_settings_dict':optimizer_settings_dict1,
            'hyper_param_dict':hyper_paramdict1,
            'modeldict':modeldict1
            } 
        
        
        newoptimizedict1=self.do_dict_override(optimizedict1,opt_dict_override,verbose=0)
        
        newhyper_paramdict1=self.build_hyper_param_start_values(newoptimizedict1['modeldict'])
        newoptimizedict1['hyper_param_dict']=newhyper_paramdict1
        try: 
            start_val_override_dict=opt_dict_override['hyper_param_dict']
            print(f'start_val_override_dict:{start_val_override_dict}')
            start_override_opt_dict={'hyper_param_dict':start_val_override_dict}
            newoptimizedict1=self.do_dict_override(newoptimizedict1,start_override_opt_dict,verbose=0)
        except:
            pass
        #    self.logger.exception(f'error in {__name__}')             
        #    print('------no start value overrides encountered------')
        #print(f'newoptimizedict1{newoptimizedict1}')
        return newoptimizedict1

