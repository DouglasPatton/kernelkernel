import os
import geopandas as gpd
import pandas as pd
import numpy as np
from pi_results import PiResults
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from helpers import Helper
from mylogger import myLogger

class Mapper(myLogger):
    def __init__(self):
        myLogger.__init__(self,name='Mapper.log')
        cwd=os.getcwd()
        self.geo_data_dir=os.path.join(cwd,'geo_data')
        self.print_dir=os.path.join(cwd,'print')
        self.boundary_data_path=self.boundaryDataCheck()
        self.boundary_dict={}
        self.pr=PiResults()
        
    def boundaryDataCheck(self):
        #https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/WBD%20v2.3%20Model%20Poster%2006012020.pdf
        datalink="https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip"
        boundary_data_path=os.path.join(self.geo_data_dir,'WBD_National_GDB.gdb')
        if not os.path.exists(boundary_data_path):
            assert False, f"cannot locate boundary data. download from {datalink}"
        return boundary_data_path
    
    def getHucBoundary(self,huc_level):
        level_digits=huc_level[-2:]
        if level_digits[0]=='0':
            level_digits=level_digits[1]
        self.boundary_dict[huc_level]=gpd.read_file(self.boundary_data_path,layer=f'WBDHU{level_digits}')
        
    def hucBoundaryMerge(self,data_df,right_on='HUC12'):
        #if right_on.lower()=='huc12':
        huc_level=right_on.lower()
        merge_kwargs={'left_on':huc_level,'right_on':right_on}
        boundary=self.getHucBoundary(huc_level)
        try: self.boundary_dict[huc_level]
        except: self.getHucBoundary(huc_level)
        return self.boundary_dict[huc_level].merge(data_df,left_on=huc_level,right_on=right_on)
  
    def plot_top_features(self,split='huc2',top_n=(-5,5),rebuild=0):
        wtd_coef_df=self.build_wt_comid_feature_importance(rebuild=rebuild)
        if split[:3]=='huc':
            split=int(split[3:])
            idx1=wtd_coef_df.index
            idx_add=[i[-2][:split] for i in idx1] # idx1 is a list of tuples 
            ##'species','estimator','HUC12','COMID', so [-2]  is 'HUC12'
            idx2=[(idx_add,*idx1[i]) for i in range(len(idx1))] # prepend idx_add
        wtd_coef_df2=wtd_coef_df.reindex(index=idx2)
        split_coef_rank=wtd_coef_df2.sum()
            
            

    def build_wt_comid_feature_importance(self,rebuild=0,fit_scorer='f1_micro'):#,wt_kwargs={'norm_index':'COMID'}):
        #get data
        coef_df=self.pr.build_spec_est_coef_df(rebuild=rebuild).drop('zzzno fish',level='species')
        spec_est_fitscor_df=self.pr.spec_est_scor_df_from_dict(
            rebuild=rebuild,scorer=fit_scorer).drop('zzzno fish',level='species')
        #get data and create accuracy measure for comid oriented weights
        y,yhat,diff=self.build_prediction_and_error_dfs(rebuild=rebuild)
        diff=diff.drop('zzzno fish',level='species')
        abs_diffscor=diff.abs().mean(axis=1).sub(1).mul(-1).to_frame()
        self.abs_diffscor=abs_diffscor
        #normalize coefficients from each model to sum to 1 for combining across estimators
        self.coef_df=coef_df
        #mod_norm_coef_df=coef_df.divide(coef_df.sum(axis=1)) # do before collapsing across estimators instead
        self.spec_est_fitscor_df=spec_est_fitscor_df
        # normalize fit scores across cv iterations
        norm_spec_est_fitscor_df=spec_est_fitscor_df.divide(spec_est_fitscor_df.sum(axis=1),axis=0)
        
        #weight coefs by relative cv score
        #no align should be necessary since fit_scor correspond to coefficients 1:1..HAHAHA
        coef_df,norm_spec_est_fitscor_df=coef_df.align(
            norm_spec_est_fitscor_df,axis=0,join='inner')#drop any unmatched rows
        
        #manually align cv_i's across multiindex columns
        fitscore_wt_coef=coef_df.copy()
        for cv_i in norm_spec_est_fitscor_df.columns: #iterating over cv_i's
            fitscore_wt_coef.loc[slice(None),(slice(None),cv_i)]=coef_df.loc[slice(None),(slice(None),cv_i)]*norm_spec_est_fitscor_df.loc[slice(None),cv_i].to_numpy()[:,None]
            #[:,None] broadcasts the numpy array of n(axis0) fitscors to all 270 columns
        cvscor_norm_coef_df=fitscore_wt_coef.sum(axis=1,level='x_var')
        self.cvscor_norm_coef_df=cvscor_norm_coef_df
        #cvscor_wtd_coef_df=coef_df.multiply(norm_spec_est_fitscor_df,axis=0)
        #cvscor_norm_coef_df=cvscor_wtd_coef_df.sum(level='cv_i')
        

        # "broadcast" scors across huc12/comids 
        cvscor_norm_coef_df_a,abs_diffscor_a=cvscor_norm_coef_df.align(
            abs_diffscor,axis=0)
        #cvscor_norm_coef_df_a.dropna(axis=1,inplace=True) #remove columns added by align
        #abs_diffscor_a.dropna(axis=1,inplace=True)
        self.cvscor_norm_coef_df_a=cvscor_norm_coef_df_a
        self.abs_diffscor_a=abs_diffscor_a
        
        # normalize abs diff scor (diff=err=y-yhat) across predictions for each comid
        
        abs_diffscor_normalized_weights=abs_diffscor/abs_diffscor.sum(axis=0,level='COMID') 
        self.abs_diffscor_normalized_weights=abs_diffscor_normalized_weights
        abs_diffscor_normalized_weights_series=abs_diffscor_normalized_weights.loc[slice(None),0]
        comid_cvscor_norm_coef_df=cvscor_norm_coef_df_a.multiply(abs_diffscor_normalized_weights_series.to_numpy(),axis=0)
        return comid_cvscor_norm_coef_df
    
    
        '''old approach below:
        coef_df=coef_df.mean(axis=1,level='x_var')
        comid_wts=self.make_comid_weights(rebuild=rebuild,norm_index=norm_index)
        coefs_wts=coef_df.join(comid_wts)
        self.coefs_wts=coefs_wts
        wtd_coef_df=coefs_wts.iloc[:,:-1].mul(coefs_wts.iloc[:,-1],axis=0)
        return wtd_coef_df'''
       
    '''def make_comid_weights(self,rebuild=0,norm_index='COMID',cv_mean=True):
        #norm_index is scope of normalization of wts (sum to 1). 
        ##e.g., for 'COMID', all specs,est wts are summed.
        spec_est_fitscor_df=self.pr.spec_est_scor_df_from_dict(rebuild=rebuild,scorer='f1_micro')
        spec_est_fitscor_df=spec_est_fitscor_df.drop('zzzno fish',level='species')
        y,yhat,diff=self.build_prediction_and_error_dfs(rebuild=rebuild)
        diff=diff.drop('zzzno fish',level='species')
        if cv_mean:
            _scor=spec_est_fitscor_df.mean(axis=1).to_frame()
            print(_scor)
            _abs_diffscor=diff.abs().mean(axis=1).sub(1).mul(-1).to_frame() # still 0 to 1, but higher is better    
        else:
            _scor=spec_est_fitscor_df
            _abs_diffscor=diff.abs().sub(1).mul(-1) # still 0 to 1, but higher is better
        self._scor=_scor
        
        self.logger.info(f'_scor:{_scor},diff:{_abs_diffscor}')
        #_scor_a=_scor.reindex(_abs_diffscor.index,axis=0 )
        #diff_a=_abs_diffscor
        _scor_a,diff_a=_scor.align(_abs_diffscor,broadcast_axis=0,join='outer',axis=0) 
        _scor_a=_scor_a.dropna(axis=1)
        diff_a=diff_a.dropna(axis=1)
        self.diff_a=diff_a
        self._scor_a=_scor_a
        if cv_mean:
            _scor_X_abs_diffscor=_scor_a.multiply(diff_a)
        else:
            cv_reps=diff_a.shape[1]
            cv_splits=int(_scor_a.shape[1]/cv_reps)
            assert _scor_a.shape[1]/cv_reps==cv_splits,'expecting int for cv_splits:{cv_splits},   diff_a.shape:{diff_a.shape},  scor_a.shape:{scor_a.shape}'
            _scor_X_abs_diffscor=diff_a.copy() #since they have the same size...
            for r in range(cv_reps):
                _scor_X_abs_diffscor.iloc[:,r]=diff_a.iloc[:,r].multiply(_scor_a.iloc[:,slice(r*cv_splits,(r+1)*cv_splits)].mean(axis=1))
        self._scor_X_abs_diffscor=_scor_X_abs_diffscor
        norm_divisor=_scor_X_abs_diffscor.sum(axis=0,level=norm_index)
        norm_divisor_a,_scor_X_abs_diffscor_a=norm_divisor.align(_scor_X_abs_diffscor,axis=0)
        
        self.norm_divisor=norm_divisor_a
        norm_scor_X_abs_diff_comid_wts=_scor_X_abs_diffscor.divide(norm_divisor_a)
        self.norm_scor_X_abs_diff_comid_wts=norm_scor_X_abs_diff_comid_wts
        frame_name='est-scorXcom-scor_wt'
        if not type(norm_scor_X_abs_diff_comid_wts) is pd.DataFrame:
            norm_scor_X_abs_diff_comid_wts=pd.DataFrame(norm_scor_X_abs_diff_comid_wts,columns=[frame_name])
        else:
            norm_scor_X_abs_diff_comid_wts.columns=[frame_name]
        return norm_scor_X_abs_diff_comid_wts'''
        
    def add_huc2_conus(self,ax):
        try: huc2=self.boundary_dict['huc02']
        except: 
            self.getHucBoundary('huc02')
            huc2=self.boundary_dict['huc02']
        huc2_conus=huc2.loc[huc2.loc[:,'huc2'].astype('int')<19,'geometry']
        huc2_conus.boundary.plot(linewidth=1,color=None,edgecolor='k',ax=ax)
    
    def plot_features_huc12(self,rebuild=0,norm_index='COMID'):
        
        wtd_coef_df=self.build_wt_comid_feature_importance(rebuild=rebuild,norm_index=norm_index)
        if norm_index=='COMID':
            wtd_coef_df.sum(axis=0,level=norm_index)
            huc12_coef_df=wtd_coef_df.mean(axis=0,level='HUC12') # sum b/c normalized 
        elif norm_index.lower()=='huc12':
            huc12_coef_df=wtd_coef_df.mean(axis=0,level='HUC12')
        else: assert False,f'unexpected norm_index:{norm_index}'
        self.huc12_coef_df=huc12_coef_df
        
        columns=list(huc12_coef_df.columns)
        colsort=np.argsort(huc12_coef_df.mean(axis=0).abs())[::-1]
        huc12_coef_gdf=self.hucBoundaryMerge(huc12_coef_df,right_on='HUC12')
        self.huc12_coef_gdf=huc12_coef_gdf
            
        fig=plt.figure(dpi=300,figsize=[10,8])
        ax=fig.add_subplot(1,1,1)
        

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
       
        huc12_coef_gdf.plot(column=columns[colsort[0]],ax=ax,cax=cax,legend=True)#,legend_kwds={'orientation':'vertical'})
        self.add_huc2_conus(ax)
        fig.savefig(Helper().getname(os.path.join(self.print_dir,'huc12_features.png')))

    
    def build_prediction_and_error_dfs(self,rebuild=0):
        agg_prediction_spec_df=self.pr.build_aggregate_predictions_by_species(rebuild=rebuild)
        yhat=agg_prediction_spec_df.drop('y_train',level='estimator').loc[:,agg_prediction_spec_df.columns!='y']
        y=agg_prediction_spec_df.xs('y_train',level='estimator').loc[:,['y']]
        y_a,yhat_a= y.align(yhat,axis=0)
        diff=-1*yhat_a.sub(y_a['y'],axis=0)
        diff.columns=[f'err_{i}' for i in range(diff.shape[1])]
        return y_a,yhat_a,diff
    
    
    def draw_huc12_truefalse(self,rebuild=0):
        try: self.boundary_dict['huc12']
        except: self.getHucBoundary('huc12')
        y,yhat,diff=self.build_prediction_and_error_dfs(rebuild=rebuild)
        nofish_diff=diff.xs('zzzno fish',level='species').copy(deep=True)
        diff=diff.drop('zzzno fish',level='species')
        
        abs_diff=diff.abs()
        nofish_abs_diff=nofish_diff.abs()
        
        err_geo_df=self.mean_flatten_and_align_huc12(diff)
        nofish_err_geo_df=self.mean_flatten_and_align_huc12(nofish_diff)
        abs_diff_geo_df=self.mean_flatten_and_align_huc12(abs_diff)
        nofish_abs_diff_geo_df=self.mean_flatten_and_align_huc12(nofish_abs_diff)
        
        
        abs_diff_geo_df.loc[:,'err']=-1*abs_diff_geo_df.loc[:,'err'].sub(1)
        rvrs_abs_diff_geo_df=abs_diff_geo_df
        nofish_abs_diff_geo_df.loc[:,'err']=-1*nofish_abs_diff_geo_df.loc[:,'err'].sub(1)
        rvrs_nofish_abs_diff_geo_df=nofish_abs_diff_geo_df
        fig=plt.figure(dpi=300,figsize=[10,8])
        ax=fig.add_subplot(2,1,1)


        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        err_geo_df.plot(column='err',ax=ax,cax=cax,legend=True,cmap='brg',)#,legend_kwds={'orientation':'vertical'})
        ax2=fig.add_subplot(2,1,2)
        nofish_err_geo_df.plot(column='err',ax=ax2,cax=cax,legend=True,cmap='brg',)
        
        
        fig.savefig(Helper().getname(os.path.join(self.print_dir,'error_map_with_zzzno-fish.png')))

            
    
    def mean_flatten_and_align_huc12(self,err_df):
        mean_err=err_df.mean(axis=1).mean(level='HUC12')
        err_df=mean_err.rename('err').reset_index()#.rename(columns={'HUC12':'huc12'})# reset_index moves index to columns for merge, rename b/c no name after the mean.
        geo_err_df=self.hucBoundaryMerge(err_df,right_on='HUC12')
        return geo_err_df
