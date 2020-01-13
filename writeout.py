import yaml

class Writer:
    def __init__(self,format='yaml'):
        self.format=format
        self.data=self.getdata()
        self.writeoutdata(format,self.data):
    
    
    def wreitoutdata(self,format,data):
        if format=='yaml':
            with open('datawrite.yaml') as f:
                yaml.dump(data,f)
        

        
    def getdata(self,):
        modeldict1={
            'loss_function':'mse',
            'Ndiff_type':'product',
            'param_count':param_count,
            'Ndiff_start':Ndiff_start,
            'max_bw_Ndiff':max_bw_Ndiff,
            'normalize_Ndiffwtsum':'own_n',
            'NWnorm':'across',
            'xkern_grid':'no',
            'ykern_grid':61,
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
        #hyper_paramdict1=self.build_hyper_param_start_values(modeldict1)
        hyper_paramdict1={}
        
        #optimization settings for Nelder-Mead optimization algorithm
        optiondict_NM={
            'xatol':0.5,
            'fatol':1,
            'adaptive':True
            }
        optimizer_settings_dict1={
            'method':'Nelder-Mead',
            'options':optiondict_NM,
            'help_start':1,
            'partial_match':1
            }
        
        optimizedict1={
            'opt_settings_dict':optimizer_settings_dict1,
            'hyper_param_dict':hyper_paramdict1,
            'modeldict':modeldict1
            } 
        outdata=[modeldict1,hyper_paramdic1,optiondict_NM,optimizer_settings_dict1,optimizedict1]
        return outdata


        
