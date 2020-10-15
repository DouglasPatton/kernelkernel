import os
import geopandas as gpd
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
        self.pr=PiResults()
        
    def boundaryDataCheck(self):
        #https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/WBD%20v2.3%20Model%20Poster%2006012020.pdf
        datalink="https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip"
        boundary_data_path=os.path.join(self.geo_data_dir,'WBD_National_GDB.gdb')
        if not os.path.exists(boundary_data_path):
            assert False, f"cannot locate boundary data. download from {datalink}"
        return boundary_data_path
    
    def getHuc12Boundary(self):
        self.h12_boundary=gpd.read_file(self.boundary_data_path,layer='WBDHU12')
    
       
    def make_weights(self,rebuild=0):
        spec_est_scor_df=self.pr.spec_est_scor_df_from_dict(rebuild=rebuild,scorer='f1_micro')
       
    
    
        y,yhat,diff=agg_prediction_spec_df=self.build_prediction_and_error_dfs(rebuild=rebuild)
        self.diff=diff
        self.logger.info(f'spec_est_scor_df:{spec_est_scor_df},diff:{diff}')
        spec_est_scor_df_a,diff_a=spec_est_scor_df.align(diff,broadcast_axis=0,join='outer')
        spec_est_scor_df_a=spec_est_scor_df_a.dropna(axis=1)
        diff_a=diff_a.dropna(axis=1)
        mean_scor=spec_est_scor_df_a.mean(axis=1)
        mean_abs_diff=diff_a.abs().mean(axis=1)
        mean_scor_X_mean_abs_diff=mean_scor.multiply(mean_abs_diff)
        mean_scor_X_mean_abs_diff.reindex(index=['species','HUC12','COMID','estimator'])
        #geo_y=self.huc12merge(y)
        #geo_yhat=self.huc12merge(yhat)
        #geo_diff=self.huc12merge(diff)
        ##self.geo_diff=geo_diff
        #geo_diff_sq=geo_diff.pow(2)
        
        
        
    def huc12merge(self,data_df,right_on='HUC12'):
        try: self.h12_boundary
        except: self.getHuc12Boundary()
        return self.h12_boundary.merge(data_df,left_on='huc12',right_on=right_on)
    
    def build_prediction_and_error_dfs(self,rebuild=0):
        agg_prediction_spec_df=self.pr.build_aggregate_predictions_by_species(rebuild=rebuild)
        yhat=agg_prediction_spec_df.drop('y_train',level='estimator').loc[:,agg_prediction_spec_df.columns!='y']
        y=agg_prediction_spec_df.xs('y_train',level='estimator').loc[:,['y']]
        y_a,yhat_a= y.align(yhat,axis=0)
        diff=-1*yhat_a.sub(y_a['y'],axis=0)
        diff.columns=[f'err_{i}' for i in range(diff.shape[1])]
        return y_a,yhat_a,diff
    
    
    def draw_huc12_truefalse(self,rebuild=0):
        try: self.h12_boundary
        except: self.getHuc12Boundary()
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
        geo_err_df=self.huc12merge(err_df,right_on='HUC12')
        return geo_err_df
