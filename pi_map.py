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
  
    def plot_top_features(self,split=None,top_n=10,rebuild=0):
        wtd_coef_df=self.build_wt_comid_feature_importance(rebuild=rebuild)
        if split is None:
            cols=wtd_coef_df.columns
            for col in cols:
                if col[:2]=='Tm':
                    wtd_coef_df.drop(col,axis=1,inplace=True)
            big_mean=wtd_coef_df.mean(axis=0)
            big_cols_sorted=list(big_mean.sort_values().index)[::-1]#descending
            big_top_2n=(big_cols_sorted[:top_n],
                        big_cols_sorted[-top_n:])
            top_pos_coef_df=wtd_coef_df.loc[:,big_top_2n[0]]
            top_neg_coef_df=wtd_coef_df.loc[:,big_top_2n[1]]
            #self.big_top_wtd_coef_df=big_top_wtd_coef_df
            pos_sort_idx=np.argsort(top_pos_coef_df,axis=1).iloc[:,::-1]
            neg_sort_idx=np.argsort(top_pos_coef_df,axis=1).iloc[:,::-1]
            #select the best and worst columns
            big_top_cols_pos=pos_sort_idx.apply(
                lambda x: big_top_2n[0][x[0]],axis=1)
            big_top_cols_neg=neg_sort_idx.apply(
                lambda x: big_top_2n[1][x[-1]],axis=1)
            colname='top_predictive_variable'
            big_top_cols_pos=big_top_cols_pos.rename(colname)
            big_top_cols_neg=big_top_cols_neg.rename(colname)
            self.big_top_cols_pos=big_top_cols_pos
            self.big_top_cols_neg=big_top_cols_neg
            
            geo_pos_cols=self.hucBoundaryMerge(big_top_cols_pos)
            geo_neg_cols=self.hucBoundaryMerge(big_top_cols_neg)
               
            fig=plt.figure(dpi=300,figsize=[10,14])
            ax=fig.add_subplot(2,1,1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            geo_pos_cols.plot(column=colname,ax=ax,cax=cax,legend=True)#,legend_kwds={'orientation':'vertical'})
            self.add_huc2_conus(ax)
            
            ax=fig.add_subplot(2,1,2)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            geo_neg_cols.plot(column=colname,ax=ax,cax=cax,legend=True)#,legend_kwds={'orientation':'vertical'})
            self.add_huc2_conus(ax)
    
            
            fig.savefig(Helper().getname(os.path.join(self.print_dir,f'huc12_top{top_n}_features.png')))

            
            
            return
        elif split[:3]=='huc': # add a new index level on lhs for huc2,etc
            split=int(split[3:])
            idx1=wtd_coef_df.index
            idx_add=[i[-2][:split] for i in idx1] # idx1 is a list of tuples 
            ##'species','estimator','HUC12','COMID', so [-2]  is 'HUC12'
            idx2=[(idx_add,*idx1[i]) for i in range(len(idx1))] # prepend idx_add
        expanded_midx=pd.MultiIndex.from_tuples(idx2,names=(split,*idx1.names))
        wtd_coef_df2=wtd_coef_df.reindex(index=expanded_midx)
        split_coef_mean=wtd_coef_df2.mean(axis=0,level=split) # mean across huc
        split_coef_rank=np.argsort(split_coef_mean,axis=1)
        
        
        #bestn=split_coef_mean.
        #bestn
            
    def normalize_and_sum_to_huc12(self,wtd_coef_df):
        wt_coef_df_sum_a,_=wt_coef_df.sum(
            axis=0,level=['species','HUC12','COMID']).align(wt_coef_df,axis=0)
        est_norm=wt_coef_df/wt_coef_df_sum_a
        estnorm_wtd_coef_df=wtd_coef_df.multiply(
            est_norm,axis=0).sum(axis=0,level=['species','HUC12','COMID'])
        estnorm_wtd_coef_df_sum_a,_=estnorm_wtd_coef_df.sum(
            axis=0,level=['HUC12','COMID']).align(estnorm_wtd_coef_df,axis=0)
        spec_norm=estnorm_wtd_coef_df/estnorm_wtd_coef_df_sum_a
        specnorm_estnorm_wtd_coef_df=estnorm_wtd_coef_df*spec_norm.sum(axis=0,level=['HUC12','COMID'])
        
        
        
        

    def build_wt_comid_feature_importance(self,rebuild=0,fit_scorer='f1_micro'):#,wt_kwargs={'norm_index':'COMID'}):
        #get data
        
        name='wt_comid_feature_importance'
        if False:#not rebuild: #just turning off rebuild here
            try:
                saved_data=self.pr.getsave_postfit_db_dict(name)
                return saved_data['data']#sqlitedict needs a key to pickle and save an object in sqlite
            except:
                self.logger.info(f'rebuilding {name} but rebuild:{rebuild}')
        datadict=self.pr.stack_predictions(rebuild=rebuild)
        y=datadict['y'];yhat=datadict['yhat']
        coef_scor_df=datadict['coef_scor_df']
        scor_indices=[]
        coef_indices=[]
        coef_scor_cols=coef_scor_df.columns
        for i,tup in enumerate(coef_scor_cols):
            if tup[0][:7]=='scorer:':
                scor_indices.append(i)
            else:
                coef_indices.append(i)
        scor_df=coef_scor_df.iloc[:,scor_indices]  
        coef_df=coef_scor_df.iloc[:,coef_indices]  
        coef_df=coef_df.dropna(axis=0,how='all')
        self.logger.info(f'BEFORE drop coef_df.shape:{coef_df.shape},scor_df.shape:{scor_df.shape}')
        coef_df,scor_df=coef_df.align(scor_df,axis=0,join='inner')
        self.logger.info(f'AFTER drop coef_df.shape:{coef_df.shape},scor_df.shape:{scor_df.shape}')
        
        #get the scorer and align to differences
        scor_select=scor_df.loc[:,('scorer:'+fit_scorer,slice(None),slice(None))]
        yhat_a,y_a=yhat.align(y,axis=0)
        
        #scor_norm=scor_select.divide(scor_select.sum(axis=1,level='var'),axis=0)
        #scor_norm2=scor_norm1.divide(scor_norm1.sum(axis=0,level='species'),axis=0)
        #wt_coef_df=coef_df.multiply(scor_norm,axis=0)
        #wt_coef_df.sum(axis=1,level=['var']) #
        #wt_coef_df.sum(axis=0,level=['species'])#after broadcasting across hucs
        ##############
        ##############      
        assert False,'halt'
        coef_df=self.pr.build_spec_est_coef_df(rebuild=rebuild).drop('zzzno fish',level='species')
        spec_est_fitscor_df=self.pr.spec_est_scor_df_from_dict(
            rebuild=rebuild,scorer=fit_scorer).drop('zzzno fish',level='species')
        #get data and create accuracy measure for comid oriented weights
        y,yhat,diff=self.build_prediction_and_error_dfs(rebuild=rebuild)
        diff=diff.drop('zzzno fish',level='species')
        abs_diffscor=diff.abs().mean(axis=1).sub(1).mul(-1)
        self.abs_diffscor=abs_diffscor
        #normalize coefficients from each model to sum to 1 for combining across estimators
        self.coef_df=coef_df
        #mod_norm_coef_df=coef_df.divide(coef_df.sum(axis=1)) # do before collapsing across estimators instead
        self.spec_est_fitscor_df=spec_est_fitscor_df
        # normalize fit scores across cv iterations
        norm_spec_est_fitscor_df=spec_est_fitscor_df.divide(spec_est_fitscor_df.sum(axis=1),axis=0)
        self.norm_spec_est_fitscor_df=norm_spec_est_fitscor_df
        #weight coefs by relative cv score
        #align on indices(0) then cv_i of multiindex columns
        coef_df_a0,norm_spec_est_fitscor_df_a0=coef_df.align(
            norm_spec_est_fitscor_df,axis=0,join='inner')#drop any unmatched rows
        coef_df_a01,norm_spec_est_fitscor_df_a01=coef_df_a0.align(
            norm_spec_est_fitscor_df_a0,axis=1,level='cv_i')
        fitscore_wt_coef=coef_df_a01.multiply(
            norm_spec_est_fitscor_df_a01,axis=0)
        
        '''fitscore_wt_coef=coef_df.copy()
        #manually align cv_i's across multiindex columns
        for cv_i in norm_spec_est_fitscor_df.columns: #iterating over cv_i's
            cvi_coef=coef_df.loc[slice(None),(slice(None),cv_i)]
            cvi_fitscor=norm_spec_est_fitscor_df.loc[slice(None),cv_i].to_numpy()[:,None]
            cvi_wtd_coef=cvi_coef.multiply(cvi_fitscor,axis=0)
            fitscore_wt_coef.loc[slice(None),(slice(None),cv_i)]=cvi_wtd_coef
            #[:,None] broadcasts the numpy array of n(axis0) fitscors to all 270 columns'''
        
        
        #sum over cv_i's for each x_var
        cvscor_norm_coef_df=fitscore_wt_coef.sum(axis=1,level='x_var')
        self.cvscor_norm_coef_df=cvscor_norm_coef_df
        

        # "broadcast" scors across huc12/comids 
        cvscor_norm_coef_df_a,abs_diffscor_a=cvscor_norm_coef_df.align(
            abs_diffscor,axis=0)
        #cvscor_norm_coef_df_a.dropna(axis=1,inplace=True) #remove columns added by align
        #abs_diffscor_a.dropna(axis=1,inplace=True)
        #self.cvscor_norm_coef_df_a=cvscor_norm_coef_df_a
        #self.abs_diffscor_a=abs_diffscor_a
        
        # normalize abs diff scor (diff=err=y-yhat) across predictions for all dimensions except huc12.
        
        abs_diffscor_normalized_weights=abs_diffscor/abs_diffscor.sum(axis=0,level='HUC12') 
        #self.abs_diffscor_normalized_weights=abs_diffscor_normalized_weights
        abs_diffscor_normalized_weights_series=abs_diffscor_normalized_weights.loc[slice(None),0]
        comid_cvscor_norm_coef_df=cvscor_norm_coef_df_a.multiply(abs_diffscor_normalized_weights_series.to_numpy(),axis=0).sum(axis=0,level='HUC12')
        
        save_data={'data':comid_cvscor_norm_coef_df} 
        self.pr.getsave_postfit_db_dict(name,save_data)
        return comid_cvscor_norm_coef_df
      
        
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
        self.y_a=y_a
        self.yhat_a=yhat_a
        self.diff=diff
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
