from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#from mpl_toolkits.axes_grid1.colorbar import colorbar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib as mpl
#from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import shapely
from shapely.geometry import Point, Polygon

import joblib
from multiprocessing import Process,Queue
from time import time,sleep
import re
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from pi_results import PiResults
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from helpers import Helper
from mylogger import myLogger
from pi_mp_helper import MatchCollapseHuc12,MpHelper,BatchOverlay
from pi_cluster import SkCluster
from pi_data_predict import PiscesPredictDataTool
from traceback import format_exc
from random import shuffle



        
        
        
    

class Mapper(myLogger):
    def __init__(self,mp=False):
        super().__init__()
        #myLogger.__init__(self,name='Mapper.log')
        cwd=os.getcwd()
        self.geo_data_dir=os.path.join(cwd,'geo_data')
        self.print_dir=os.path.join(cwd,'print')
        self.species_map_dir=os.path.join(cwd,'species_maps')
        if not os.path.exists(self.species_map_dir: os.mkdir(self.species_map_dir)
        self.states_path=os.path.join(self.geo_data_dir,'states','cb_2017_us_state_500k.dbf')
        self.boundary_data_path=self.boundaryDataCheck()
        #self.NHD_data_path=self.nhdDataCheck()
        self.NHDPlusV21_data_path=self.NHDPlusV21DataCheck()
        self.NHDPlusV21_CatchmentSP_data_path=os.path.join(self.geo_data_dir,'NHDPlusV21_CatchmentSP.feather')
        #NHDplus URL:  https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/NHDPLUS_H_0101_HU4_GDB.zip   
        self.boundary_dict={}
        self.cv_pr=PiResults(cv_run=True)
        self.fit_scorer=self.cv_pr.fit_scorer
        plt.rc_context({
            
            'figure.autolayout': True,
            'axes.edgecolor':'k', 
            'xtick.color':'k', 'ytick.color':'k', 
            'figure.facecolor':'grey'})
        if mp:
            self.set_states()
            self.setNHDPlusV21CatchmentSP()
            self.ppdt=PiscesPredictDataTool(cv_run=False)
            self.ppdt.setgt()
            try:self.huc12comiddict
            except:self.huc12comiddict={**self.ppdt.gt.gethuc12comiddict()}
            try: self.huc8_huc12dict
            except:self.huc8_huc12dict={**self.ppdt.gt.build_huchuc()['huc8_huc12dict']}
        self.asers=[];self.psers=[] # for debugging
    
    def NHDPlusV21DataCheck(self):
        # data guide: https://s3.amazonaws.com/edap-nhdplus/NHDPlusV21/Data/NationalData/0Release_Notes_NationalData_Seamless_GeoDatabase.pdf
        
        datalink='https://s3.amazonaws.com/edap-nhdplus/NHDPlusV21/Data/NationalData/NHDPlusV21_NationalData_Seamless_Geodatabase_Lower48_07.7z'
        datapath=os.path.join(self.geo_data_dir,'NHDPlusV21','NHDPlusNationalData','NHDPlusV21_National_Seamless_Flattened_Lower48.gdb')
        if not os.path.exists(datapath):
            print(f'NHDPlusV21 not detected. available at {datalink}')
            assert False,'Halt'
        else: return datapath
    def set_states(self):
        states=gpd.read_file(self.states_path)
        fips_ordered,ilocs=zip(*sorted(zip(states.STATEFP.to_list(),list(range(len(states))))))
        ilocs=list(ilocs);fips_ordered=list(fips_ordered)
        alaska_idx=fips_ordered.index('02')
        ilocs.pop(alaska_idx)
        fips_ordered.pop(alaska_idx)
        hawaii_idx=fips_ordered.index('15')
        ilocs.pop(hawaii_idx)
        fips_ordered.pop(hawaii_idx)
        self.states=states.iloc[ilocs[:49]]
        
    def add_states(self,ax,clip_to=None,bbox=None,zorder=9,crs=None,plot_kwargs={}):
        self.logger.info('adding states')
        try: self.states
        except: self.set_states()
            
        if not clip_to is None:
            bounds=clip_to.total_bounds
            states=self.states.cx[bounds[0]:bounds[2],bounds[1]:bounds[3]]
        elif not bbox is None:
            #states=self.states.cx[bbox[0]:bbox[2],bbox[1]:bbox[3]]
            states=gpd.overlay(self.states,self.gdfBoxFromOuterBounds(bbox,crs),how='intersection')
        else:
            states=self.states
        self.logger.info(f'states.total_bounds: {states.total_bounds}')
        states.plot(ax=ax,color='wheat',zorder=zorder)  
        states.boundary.plot(linewidth=0.2,ax=ax,edgecolor='k',zorder=8,**plot_kwargs)  
        
    def withinBoundsCheck(self,i_b,o_b):
        #i_b for inner bounds to check and o_b for outer bounds
        is_within=[True]*4
        if i_b[0]<o_b[0]:is_within[0]=False
        if i_b[1]<o_b[1]:is_within[1]=False
        if i_b[2]>o_b[2]:is_within[2]=False
        if i_b[3]>o_b[3]:is_within[3]=False
        if all(is_within):
            return True
        else:
            self.logger.info(f'clipping required. is_within:{is_within} for inner_bounds: {i_b} and outer_bounds: {o_b}')
            return False
    def expandBBox(self,bbox,ratio,extra_on_top=0):
        
        
        width=abs(bbox[2]-bbox[0])*(ratio-1)
        height=abs(bbox[3]-bbox[1])*(ratio-1)
        expansion_factor=min(width,height)
        bigger_bbox=(bbox[0]-expansion_factor/2,bbox[1]-expansion_factor/2,bbox[2]+expansion_factor/2,bbox[3]+expansion_factor/2)
        if extra_on_top>0:
            top=bigger_bbox[3]+(expansion_factor)*(1+extra_on_top)
            bigger_bbox=(*bigger_bbox[:3],top)
        return bigger_bbox
        
    def boundaryDataCheck(self):
        #https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/WBD%20v2.3%20Model%20Poster%2006012020.pdf
        datalink="https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/National/GDB/WBD_National_GDB.zip"
        boundary_data_path=os.path.join(self.geo_data_dir,'WBD_National_GDB.gdb')
        if not os.path.exists(boundary_data_path):
            assert False, f"cannot locate boundary data. download from {datalink}"
        return boundary_data_path
    
    def getNHDPlusV21Layer(self,layer_name):
        return gpd.read_file(self.NHDPlusV21_data_path,layer=layer_name)
    
    def setNHDPlusV21CatchmentSP(self):#SP for simplified polygons
        try: self.states
        except:self.set_states()
        if not os.path.exists(self.NHDPlusV21_CatchmentSP_data_path):
            print('resaving the catchmentsp layer',end='....')
            NHDPlusV21CatchmentSP=self.getNHDPlusV21Layer('CatchmentSP')
            print('starting nhdplusv21catchment clip')
            n=NHDPlusV21CatchmentSP.shape[0]
            procs=12
            ch=-(-n//procs)
            states_outline=self.states.dissolve()
            outlist=MpHelper().runAsMultiProc(BatchOverlay,[[NHDPlusV21CatchmentSP.iloc[ch*i:ch*(i+1)],states_outline] for i in range(procs)])
            self.logger.info('clipping done, starting concatenation')
            NHDPlusV21CatchmentSP=pd.concat([list_i for list_ii in outlist for list_i in list_ii]) #flatten list of lists of gdfs to list of gdfs
            self.logger.info(f'after concat, crs:{NHDPlusV21CatchmentSP.crs}, shape: {NHDPlusV21CatchmentSP.shape} and from before, n: {n}')
            
            #NHDPlusV21CatchmentSP=gpd.overlay(NHDPlusV21CatchmentSP,self.states.dissolve(),how='intersection')
            print('catchment clip completed')
            NHDPlusV21CatchmentSP.to_feather(self.NHDPlusV21_CatchmentSP_data_path)
            print('...is complete')
            self.NHDPlusV21CatchmentSP=NHDPlusV21CatchmentSP
        else:
            print('reading feather',end='')
            self.NHDPlusV21CatchmentSP=gpd.read_feather(self.NHDPlusV21_CatchmentSP_data_path)
            print('...is complete')
    
    def findBoxRatio(self,box1,box2,geo='area'):
        w1=box1[2]-box1[0]
        w2=box2[2]-box2[0]
        h1=box1[3]-box1[1]
        h2=box2[3]-box2[1]
        if geo=='area':return w1*h1/(w2*h2)
        elif geo=='x':return w1**1/w2**1
        elif geo=='y': return h1**1/h2**1
        else: 
            assert False, f"findBoxRatio doesn't recognize geo:{geo}"

    def addDroppedLeadingZero(self,list_of_strings):
        new_list=[]
        for string in list_of_strings:
            if len(string)==8:
                new_list.append(string)
            elif len(string)==7:
                new_list.append('0'+string)
            else:
                assert False,f'unexpected string without len 7 or 8: {string}'
        return new_list
    
            
            
    def getExtendedHuc8s(self,spec,huc8list):
        try:self.ppdt.specieshuc8list
        except:self.ppdt.buildspecieshuc8list()
        specieshuc8list=self.ppdt.specieshuc8list
        try: self.ppdt.specieshuclist
        except:self.ppdt.buildspecieslist()
        specieshuclist=self.ppdt.specieshuclist #from the survey
        specieslist=self.ppdt.specieslist
        idx=specieslist.index(spec)
        #########
        species_huc8_dict={}
        list1=specieshuclist[idx]
        list2=specieshuc8list[idx]
        list1=self.addDroppedLeadingZero(list1)
        list2=self.addDroppedLeadingZero(list2)
        biglist=list(dict.fromkeys([*list1,*list2]))
        novel_huc8_list=list(set(biglist)-set(huc8list))
        if len(novel_huc8_list)>0:
            return novel_huc8_list#self.getHucDigitsGDF(novel_huc8_list,huc='08').plot(ax=ax,zorder=4)
        else:
            return None
        
    def getNotSampledNSHuc8s(self,NS_huc8_list):
        return list(set(NS_huc8_list)-set(self.addDroppedLeadingZero(self.ppdt.huclist_survey)))

            
    def truncateExtendedHucs(self,extended_huc8_list,NS_huc8_list):
        try:
            bbox=self.getHucDigitsGDF(NS_huc8_list,huc='08').total_bounds
            selected=self.getHucDigitsGDF(extended_huc8_list,huc='08').cx[bbox[0]:bbox[2],bbox[1]:bbox[3]].loc[:,'huc8'].tolist()
            if len(selected) is None:
                return []
            else:
                return selected
        except:
            return []
            
            
    def plotSpeciesPredictList(self,species_list,species_plot_kwarg_dict):
        for species in species_list:
            self.plotSpeciesPredict(species,**species_plot_kwarg_dict)
            
          
    def plotSpeciesPredict(
        self,species,
        estimator_name=None,huc_level=2,include_absent=False,save_check=False,
        plot_train=False,include_extended_hucs='truncated',
        main_plot='classify',#'cv_probability',#'classify'
        secondary_plots={}#{'binary_classify':'middle'} # 'cv_PI-lower-5%', '
    ):
        '''if slow, try https://gis.stackexchange.com/questions/197945/geopandas-polygon-to-matplotlib-patches-polygon-conversion'''
        try:
            
            
            format_name=species[0].upper()+species[1:].lower()
            
            name=f'{main_plot}_{format_name}.png'
            if  not estimator_name is None:
                name+=f'_{estimator_name}'
            
            
            
            savepath=os.path.join(self.species_map_dir,name)
            if save_check and os.path.exists(savepath):
                print(f'{species} already saved, skipping')
                return
            
            plot_names=[main_plot, *secondary_plots.keys()]
            assert all([type(name) is str for name in plot_names])
            incl_cv_plot=any([name[:3].lower()=='cv_' for name in plot_names]) #relies on cv plots having cv_ prefix
            incl_single_fit_plot=any([name[:3].lower()!='cv_' for name in plot_names])
            try:self.NHDPlusV21CatchmentSP
            except:self.setNHDPlusV21CatchmentSP()
            crs=self.NHDPlusV21CatchmentSP.crs
            try: self.states
            except: self.set_states()
            try:self.ppdt
            except:
                self.ppdt=PiscesPredictDataTool(cv_run=False)
                self.ppdt.setgt()
            try: self.ppdt.huclist_survey
            except: self.ppdt.buildspecieslist()
            if incl_cv_plot:
                try: self.cv_ppdt
                except:self.cv_ppdt=PiscesPredictDataTool(cv_run=True)
                
            
            NS_huc8_list=self.ppdt.getSpeciesHuc8List(species,only_new_hucs=False) #natureserve(NS) huc8's
            not_sampled_NS_huc8_list=self.getNotSampledNSHuc8s(NS_huc8_list)
            #huc_range=self.getHucDigitsGDF(NS_huc8_list,huc='08')
            if len(not_sampled_NS_huc8_list)>0:
                not_sampled_huc_range=self.getHucDigitsGDF(not_sampled_NS_huc8_list,huc='08')
            else: 
                not_sampled_huc_range=None
            if include_extended_hucs:    
                extended_huc8_list=self.getExtendedHuc8s(species,NS_huc8_list)
                if extended_huc8_list:
                    if type(include_extended_hucs) is str:
                        if include_extended_hucs.lower()=='truncated':
                            extended_huc8_list=self.truncateExtendedHucs(extended_huc8_list,NS_huc8_list)
                        else: assert False, 'not developed'
                    combined_huc8_list=list(dict.fromkeys([*extended_huc8_list,*NS_huc8_list]))
                if extended_huc8_list:
                    extended_huc8_gdf=self.getHucDigitsGDF(extended_huc8_list,huc='08')
                else:
                    extended_huc8_gdf=None
            if not include_extended_hucs or not extended_huc8_list:
                extended_huc8_gdf=None
                extended_huc8_list=None
                combined_huc8_list=NS_huc8_list
            self.logger.info(f'species:{species} extended_huc8_list:{extended_huc8_list}, combined_huc8_list:{combined_huc8_list}, NS_huc8_list:{NS_huc8_list}')
            combined_huc8_range=self.getHucDigitsGDF(combined_huc8_list,huc='08')
            huc_outer_bounds=combined_huc8_range.total_bounds
            self.logger.info(f'huc_outer_bounds:{huc_outer_bounds}')
            w=huc_outer_bounds[2]-huc_outer_bounds[0]
            h=huc_outer_bounds[3]-huc_outer_bounds[1]
            geo='x'# if w<h else 'y' #for scaling inset us states map according to shorter axis
            
            huc_range_box=self.gdfBoxFromOuterBounds(huc_outer_bounds,crs)
            if max(w,h)<5:
                expansion_factor=1.3 #show more context in small maps
                max_fig_len=6
            elif max(w,h)<7.5: 
                expansion_factor=1.2
                max_fig_len=7
            
            elif max(w,h)<10:
                expansion_factor=1.1
                max_fig_len=8
            elif max(w,h)<15:
                expansion_factor=1.05
                max_fig_len=9
            else:
                expansion_factor=1.02
                max_fig_len=11
                
            if max(w,h)>20 and min(w,h)>15:
                do_inset=False
            else: do_inset=True
                
            conus_outline=self.states.dissolve()    
            buffered_huc_outer_bounds=self.expandBBox(huc_outer_bounds,expansion_factor,)#extra_on_top=(expansion_factor-1)/2)
            
            buffered_huc_range_box=self.gdfBoxFromOuterBounds(buffered_huc_outer_bounds,crs)
            
            if 1.4*h>=w:
                orientation='tall'
                fig= plt.figure(figsize=[0.3*max_fig_len+.7*max_fig_len*w/h,max_fig_len],dpi=1200,)#width adjust, but less than proportionally.
                plot_split=5
                ps2=int(plot_split*w/h)+1
                gs=fig.add_gridspec(plot_split,ps2)
                #gs.update(wspace=0.025, hspace=0.05)
                ax=fig.add_subplot(gs[:,0:ps2-1])
                if do_inset:inset_ax=fig.add_subplot(gs[0,ps2-1])
                #cbar_ax=fig.add_subplot(gs[1,ps2-1])
                cbar_ax=make_axes_locatable(ax).append_axes('top',size='5%',pad='2%')
                sample_ax=fig.add_subplot(gs[1:3,ps2-1])
                #legend_ax=fig.add_subplot(gs[3,ps2-1])
            else:
                orientation='wide'
                fig = plt.figure(figsize=[max_fig_len,.25*max_fig_len+0.75*max_fig_len*h/w],dpi=1200)#height adjust, but less than proportionally.
                plot_split=7
                ps2=int(plot_split*h/w)+1
                gs=fig.add_gridspec(ps2,plot_split,)
                #gs.update(wspace=0.025, hspace=0.05)
                ax=fig.add_subplot(gs[0:ps2-1,:])
                if do_inset:inset_ax=fig.add_subplot(gs[ps2-1,0])
                #cbar_ax=fig.add_subplot(gs[ps2-1,1])
                cbar_ax=make_axes_locatable(ax).append_axes('left',size='5%',pad=0.5)
                fig.add_axes(cbar_ax)
                sample_ax=fig.add_subplot(gs[ps2-1,1:3])
                #legend_ax=fig.add_subplot(gs[ps2-1,3])
            #fig, ax = plt.subplots(figsize=[8,8],dpi=1200)#height adjust, but less than proportionally.
            legend_patches=[];ncols=0#for building legend
            buffered_huc_range_box.plot(ax=ax,zorder=0,color='c')
            buffered_huc_range_box.plot(ax=sample_ax,zorder=0,color='c',alpha=0.2)
            plt.tick_params(axis='both',which='both',bottom=False,left=False,
                                top=False,labelbottom=False,labelleft=False)
            self.logger.info(
                f'buffered_huc_range_box.total_bounds: {buffered_huc_range_box.total_bounds}')
            huc_range_intersect=gpd.overlay(combined_huc8_range,conus_outline,how='intersection') #clip hucs to same coastal boundary as states
            huc_range_intersect.plot(ax=sample_ax,zorder=3,color='darkgrey',edgecolor=None)
            if not not_sampled_huc_range is None:
                not_sampled_huc_range_intersect=gpd.overlay(not_sampled_huc_range,conus_outline,how='intersection')
                not_sampled_huc_range_intersect.plot(ax=sample_ax,zorder=4,color='black',edgecolor='lightgray' ,
                                                     #hatch=3*'+', 
                                                     lw=0.25)
            else:
                not_sampled_huc_range_intersect=None
            
            print(f'building data for {species}')
            self.logger.info(f'building data for {species}')
            cv_data_exists=False
            if incl_cv_plot:
                try:
                    cv_huc_species_series_dict,predict_huc8_list=self.cv_ppdt.BuildBigSpeciesXPredictDF(
                        species=species,estimator_name=estimator_name,hucdigitcount=huc_level)
                    self.cv_huc_species_series_dict=cv_huc_species_series_dict
                    cv_data_exists=True
                except:
                    self.logger.exception('error trying to BuildBigSpeciesXPredictDF for cv_run, skipping')
                    #cv_data_exists=False
            if incl_single_fit_plot or not cv_data_exists:
                huc_species_series_dict,cv_predict_huc8_list=self.ppdt.BuildBigSpeciesXPredictDF(
                    species=species,estimator_name=estimator_name,hucdigitcount=huc_level)
                #self.huc_species_series_dict=huc_species_series_dict
            try: predict_huc8_list
            except: predict_huc8_list=cv_predict_huc8_list
            """
            missing_data_comids=self.buildMissingComids(
                predict_huc8_list,
                huc_species_series_dict if main_plot=='classify' else cv_huc_species_series_dict,
                huc_level
            )
            self.missing_data_comids=missing_data_comids
            if len(missing_data_comids)>0:
                plot_kwargs={'color':'red','edgecolor':None,'zorder':7}
                '''self.plotByHucDict([ax,sample_ax],
                    missing_data_comids,huc_outer_bounds,huc_range_box,main_plot,
                    None,plot_kwargs=plot_kwargs,
                    orientation=orientation
                    )'''
            """
            huc_range_intersect.plot(ax=ax,zorder=5,color='red',edgecolor=None)
            legend_patches.append(mpatches.Patch(facecolor='red',edgecolor=None, label='missing data'))
            ncols+=1
            
            
            
            print(f'plotting {species}...',end='')
            if main_plot=='classify':
                self.plotByHucDict(
                    ax,huc_species_series_dict,huc_outer_bounds,huc_range_box,main_plot,
                    include_absent,plot_kwargs=dict(column='y',zorder=6,color='r'),orientation=orientation)
            elif main_plot=='cv_probability':
                self.plotByHucDict(ax,
                    cv_huc_species_series_dict,huc_outer_bounds,huc_range_box,main_plot,
                    include_absent,
                    plot_kwargs=dict(column='y',zorder=6,label='cv_probability'),
                    orientation=orientation
                    )
            """
            predict_huc8_keys=dict.fromkeys(predict_huc8_list)
            #self.logger.info(f'combined_huc8_list: {combined_huc8_list}, predict_huc8_list: {predict_huc8_keys}' )
            missing_data_huc8_list=[huc for huc in combined_huc8_list if huc not in predict_huc8_keys]#faster search with dict
            if len(missing_data_huc8_list)>0:
                missing_data_gdf=gpd.overlay(self.getHucDigitsGDF(missing_data_huc8_list,huc='08'),buffered_huc_range_box,how='intersection')
                if missing_data_gdf.size>0:
                    missing_data_gdf.plot(ax=ax,zorder=15,color='red',edgecolor=None,)"""
            
                
            
                                                                           
            self.add_states(ax,bbox=buffered_huc_outer_bounds,zorder=2,crs=crs)
            self.add_states(sample_ax,bbox=buffered_huc_outer_bounds,zorder=2,crs=crs,plot_kwargs=dict(alpha=0.3))
            #huc_range=gpd.overlay(combined_huc8_range,conus_outline, how='intersection')
            huc_range_intersect.boundary.plot(ax=sample_ax, zorder=10,linewidth=0.25, edgecolor='black')
            #huc_range.plot(ax=ax, zorder=10,linewidth=0.25,color='k' color='black')    
            if not extended_huc8_gdf is None:
                extended_huc8_gdf_intersect=gpd.overlay(extended_huc8_gdf,conus_outline,how='intersection')
                extended_huc8_gdf_intersect.boundary.plot(ax=ax,zorder=11,color='orange',linewidth=0.5) 
                extended_huc8_gdf_intersect.plot(ax=sample_ax,zorder=11,color='orange',linewidth=1) 
                plt.tick_params(axis='both',which='both',bottom=False,left=False,
                                top=False,labelbottom=False,labelleft=False)
            ax.set_aspect('equal')
            sample_ax.set_aspect('equal')
            
            #if self.plot_train:
            if do_inset:
                #help from https://jeremysze.github.io/GIS_exploration/build/html/zoomed_inset_axes.html
                expanded_states_bounds=self.expandBBox(self.states.total_bounds,1.15)
                g='x' if w<h else 'y'
                r=self.findBoxRatio(buffered_huc_outer_bounds,expanded_states_bounds,geo=g)
                #ag=np.log((-np.log(r)))/40
                mag=r*0.15
                #inset_ax = zoomed_inset_axes(ax, mag, loc=2)
                inset_ax.set_xlim(expanded_states_bounds[0], expanded_states_bounds[2])
                inset_ax.set_ylim(expanded_states_bounds[1], expanded_states_bounds[3])
                #self.gdfBoxFromOuterBounds(
                #    expanded_states_bounds,crs).plot(ax=inset_ax,color='c',zorder=0,alpha=0.5)
                self.states.plot(ax=inset_ax,facecolor='tan',edgecolor='none',zorder=1,alpha=0.6)
                self.states.plot(ax=inset_ax,zorder=2,linewidth=0.5,edgecolor='grey',facecolor='none')
                lw=(-np.log(r)*1.7)**.1
                buffered_huc_range_box.boundary.plot(ax=inset_ax,color='k',zorder=3,linewidth=lw,alpha=1)
                inset_ax.margins(0)
                plt.tick_params(axis='both',which='both',bottom=False,left=False,
                                top=False,labelbottom=False,labelleft=False)
                

            #format_name_parts=re.split(' ',species[0].upper()+species[1:].lower())
            format_name_parts=re.split(' ',format_name)
            title=f'Predicted Distribution for '+" ".join([f'$\it{{{part}}}$' for part in format_name_parts])
            fig.suptitle(title)  #\it destroys spaces!!
            #fig.suptitle(f'Predicted Distribution for $\it{{{format_name_parts[0]}}}$ $\it{{{format_name_parts[1]}}}$')
            #self.addInverseConus(ax,buffered_huc_outer_bounds,gdf.crs,zorder=9)
            ax.margins(0)
            if not extended_huc8_gdf is None:
                    legend_patches.append(mpatches.Patch(facecolor='orange',edgecolor='orange', label='extended range'))
                    ncols+=1
            if main_plot=='classify':
                legend_patches.append(mpatches.Patch(color='magenta', label='Present'))
                ncols+=1
                if include_absent:
                    legend_patches.append(mpatches.Patch(facecolor='cyan', label='Absent'))
                    ncols+=1
            legend_patches.append(mpatches.Patch(facecolor='darkgrey',label='in sample'))
            ncols+=1
            if not not_sampled_huc_range_intersect is None:
                legend_patches.append(mpatches.Patch(facecolor='black',
                                                     #hatch=3*'+',
                                                     label='out of sample'))
                ncols+=1
           
            if orientation=='tall':
                ncols=1# if ncols > 1 else 0
                bbta=(0,0)
                lloc='upper left'
            else:
                bbta=(1,0)
                lloc='lower left'
            if ncols>0:
                sample_ax.legend(handles=legend_patches,fontsize='x-small',ncol=ncols,bbox_to_anchor=bbta,loc=lloc)
                #fig.legend(handles=legend_patches,fontsize='x-small',ncol=ncols,loc=lloc)
            if main_plot=='cv_probability':
                #fig.subplots_adjust(bottom=0.5)
                cmap = mpl.cm.cool
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                fig.colorbar(
                    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cbar_ax, orientation='horizontal' if not orientation=='wide' else 'vertical', )
                    #label='Presence Probability', fontsize='x-small')
                cbar_ax.set_title('Presence Probability',fontsize='xx-small')
                if orientation=='tall':cbar_ax.xaxis.set_ticks_position("top")
            else:
                assert False, f'unexpected val for main_plot: {main_plot}'
            #if orientation=='tall':
            ykwarg={} if (orientation=='tall' or not do_inset) else {'y':0.9}
            sample_ax.set_title('HUC08 Range', fontsize='xx-small',**ykwarg)
            #else:
            #    sample_ax.text(0.5,0,'HUC08 Range',rotation=90,va='center', fontsize='xx-small')
            #ax.set_axis_off()
            
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            if do_inset:
                inset_ax.axes.xaxis.set_ticks([])
                inset_ax.axes.yaxis.set_ticks([])
                inset_ax.axes.xaxis.set_ticklabels([])
                inset_ax.axes.yaxis.set_ticklabels([])
            sample_ax.axes.xaxis.set_ticks([])
            sample_ax.axes.yaxis.set_ticks([])
            sample_ax.axes.xaxis.set_ticklabels([])
            sample_ax.axes.yaxis.set_ticklabels([])
            sample_ax.axis('off')
            if do_inset:inset_ax.axis('off')
            ax.axis('off')
            fig.tight_layout
            fig.show() 
            fig.savefig(savepath)
        except:
            self.logger.exception('outer catch')
            
            
    def plotByHucDict(self,ax,huc_species_series_dict,huc_outer_bounds,huc_range_box,plot_type,include_absent,plot_kwargs=dict(column='y',zorder=6,),orientation='wide'):
        if not type(huc_species_series_dict) is dict:
            huc_species_series_dict={'no_key':huc_species_series_dict}
        
        for huc,ser in huc_species_series_dict.items():
            ser_dict={};plot_kwarg_dict={}
            col_name=ser.columns[0]
            assert col_name=='y',f'expecting "y" at 1st pos, but columns:{ser.columns}' #may need more flexibility later...
            print(f'{huc}',end=', ')
            if plot_type=='classify':
                cser=ser.copy()[col_name]
                cser[:]='missing data'
                pser=cser[ser[col_name]>=0.5]
                #cser[ser.to_numpy()<0.5]='absent'
                pser[:]='present'
                self.psers.append(pser)
                #print(pser)
                if len(pser)>0:
                    ser_dict['present']=pser
                    plot_kwargs_i=plot_kwargs.copy()
                    plot_kwargs_i['color']='r'
                    plot_kwarg_dict['present']=plot_kwargs_i
                else:
                    if not include_absent:
                        continue
                if include_absent:
                    aser=cser[ser[col_name]<0.5]
                    aser[:]='absent'
                    self.asers.append(aser)
                    if len(aser)>0:
                        ser_dict['absent']=aser
                        plot_kwargs_i=plot_kwargs.copy()
                        plot_kwargs_i['color']='b'
                        plot_kwarg_dict['absent']=plot_kwargs_i
                    
            elif plot_type=='cv_probability':
                ser_dict['cv_probability']=ser
                #divider = make_axes_locatable(ax)
                plot_kwarg_dict[plot_type]={**plot_kwargs,'cmap':mpl.cm.cool,'vmin':0,'vmax':1,}
                    #'cax':divider.append_axes("right" if orientation=='wide' else 'bottom', size="5%", pad=0.1),
                    #'legend':True,'cmap':'brg'} #moved 
            else:
                ser_dict={huc:ser}
                plot_kwarg_dict={huc:plot_kwargs}
            for ser_name,ser in ser_dict.items():
                if type(ser.index) is pd.MultiIndex:
                    try:
                        ser.index=ser.index.get_level_values('COMID')
                    except:
                        assert False,f'ser.index:{ser.index}'
                if type(ser) is pd.DataFrame:
                    #if type(ser.columns) is pd.MultiIndex:
                        #ser.columns=ser.columns.get_level_values('var').tolist()
                    ser=ser.loc[:,'y']
                ser.index=ser.index.astype('int64')

                gdf=self.NHDPlusV21CatchmentSP.merge(
                    ser,how='inner',right_on='COMID',left_on='FEATUREID',right_index=True)
                if len(gdf)==0:continue
                if not self.withinBoundsCheck(gdf.total_bounds,huc_outer_bounds):
                    gdf=gpd.overlay(gdf,huc_range_box,how='intersection')
                self.logger.info(f'plotting (huc,ser_name):{(huc,ser_name)}, total_bounds: {gdf.total_bounds}')
                if not type(ax) is list:
                    ax_list=[ax]
                else:
                    ax_list=ax
                for ax_i in ax_list: gdf.plot(ax=ax_i,**plot_kwarg_dict[ser_name])
    
    def buildMissingComids(self,huc8_list,hucN_y_dict,huc_level):
        try:self.huc12comiddict
        except:self.huc12comiddict={**self.ppdt.gt.gethuc12comiddict()}
        try: self.huc8_huc12dict
        except:self.huc8_huc12dict={**self.ppdt.gt.build_huchuc()['huc8_huc12dict']}
        fixkeys=[]
        for huc8 in self.huc8_huc12dict.keys():
            if len(huc8)==7:
                fixkeys.append(huc8)
        for huc8 in fixkeys:
            val=self.huc8_huc12dict.pop(huc8)
            self.huc8_huc12dict['0'+huc8]=val    
        if huc_level<8:
            full_comid_y_dict={hucN:[] for hucN in hucN_y_dict.keys()}
            for huc8 in huc8_list:
                
                full_comid_y_dict[huc8[:huc_level]].extend(
                    [comid for c_list in [self.huc12comiddict[huc12] for huc12 in self.huc8_huc12dict[huc8]] for comid in c_list]
                )
        elif huc_level==8:
            full_comid_y_dict={huc8:[comid for c_list in [self.huc12comiddict[huc12] for huc12 in self.huc8_huc12dict[huc8]] for comid in c_list] for huc8 in huc8_list}
        
        else: assert False,f'huc_level>8 not developed, huc_level: {huc_level}'
        hucN_missing_serdict={}        
        for hucN,c_list in full_comid_y_dict.items():
            missing_c_list=list(set(c_list)-set(hucN_y_dict[hucN].index.get_level_values('COMID')))
            if len(missing_c_list)==0:
                self.logger.info(f'no missing data in hucN:{hucN}')
                continue
            midx=pd.MultiIndex.from_tuples([(c,) for c in missing_c_list],names=['COMID',])
            ser=pd.DataFrame([1]*len(missing_c_list),index=midx,columns=['y'])
            hucN_missing_serdict[hucN]=ser
        return hucN_missing_serdict
                           
                           
    
    
        
    def gdfBoxFromOuterBounds(self,outer_bounds,crs):
        bbox=outer_bounds
        p1 = Point(bbox[0], bbox[3])
        p2 = Point(bbox[2], bbox[3])
        p3 = Point(bbox[2], bbox[1])
        p4 = Point(bbox[0], bbox[1])

        np1 = (p1.coords.xy[0][0], p1.coords.xy[1][0])
        np2 = (p2.coords.xy[0][0], p2.coords.xy[1][0])
        np3 = (p3.coords.xy[0][0], p3.coords.xy[1][0])
        np4 = (p4.coords.xy[0][0], p4.coords.xy[1][0])

        bb_polygon = Polygon([np1, np2, np3, np4])
        boxgdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(bb_polygon), columns=['geometry'],crs=crs)
        return boxgdf
        
    def addInverseConus(self,ax,outer_bounds,crs,zorder=9):
        self.logger.info('adding inverse conus')
        df2=self.gdfBoxFromOuterBounds(outer_bounds,crs)
        gpd_clipped=gpd.overlay(df2,self.states,how='difference')
        gpd_clipped.plot(ax=ax,color='w',zorder=zorder)
                        
                        
    def stringifyHuc(self,huc):
        if type(huc) is int:
            huc=str(huc)
        if not len(huc)%2==0: huc='0'+huc
        #assert len(huc)==2,f'expecting 2 digit string, but huc: {huc}'
        return huc
        
    def getHucDigitsGDF(self,hlist,huc='02'):
        huc=self.stringifyHuc(huc)
        hlist=list(dict.fromkeys([self.stringifyHuc(h)[:int(huc)] for h in hlist]))#fromkeys removes duplicates
            
        try: huc_gdf=self.boundary_dict[f'huc{huc}']
        except: huc_gdf=self.getHucBoundary(f'huc{huc}')
        huc_column_name=f'huc{huc}'
        if huc_column_name[-2]=='0': #b/c column name might be huc2 not huc02
            huc_column_name=huc_column_name[:-2]+huc_column_name[-1]
        return huc_gdf.loc[huc_gdf[huc_column_name].isin(hlist)]
        
    
    def getHucBoundary(self,huc_level):
        #print(huc_level)
        level_digits=huc_level[-2:]
        if level_digits[0]=='0' or not level_digits[0].isdigit():
            level_digits=level_digits[1] #e.g. HUC02 --> 2
        return gpd.read_file(self.boundary_data_path,layer=f'WBDHU{level_digits}')
        
    def hucBoundaryMerge(self,data_df,right_on='HUC12'):
        # hucboundary files have index levels like 'huc2' not 'HUC02'
        #if right_on.lower()=='huc12':
        self.logger.info(f'starting boundary merge')
        #self.data_df=data_df
        if type(right_on) is str:
            huc_level=right_on.lower()
        elif type(right_on) is int:
            huc_level=f'huc{right_on}' #'huc' not necessry
        else:
            assert False, f'unexepected right_on:{right_on} not str or int'
        level_digits=huc_level[-2:]
        if huc_level[-2]=='0':
            huc_level=huc_level[:-2]+huc_level[-1]
        if type(data_df.index) is pd.MultiIndex:
            h_list=data_df.index.unique(level=right_on)
        else:
            h_list=list(data_df.index)
        boundary=self.getHucBoundary(huc_level)
        boundary_clip=boundary.loc[boundary[huc_level].isin(h_list)]
        #boundary_clip=boundary.loc[selector]
        merged=boundary_clip.merge(data_df,left_on=huc_level,right_on=right_on)
        self.logger.info(f'boundary merge completed')
        return merged
    
    
    def plot_y01(self,zzzno_fish=False,rebuild=0,huc_level=None):
        coef_df,scor_df,y,yhat=self.cv_pr.get_coef_stack(
            rebuild=rebuild,drop_zzz=not zzzno_fish,return_y_yhat=True,
            drop_nocoef_scors=False)
        
        h12_y1sum=y.sum(axis=0,level=['HUC12','COMID']).mean(axis=0,level='HUC12')
        h12_y1sum.columns=['species found']
        h12_y0sum=(1-y).sum(axis=0,level=['HUC12','COMID']).mean(axis=0,level='HUC12')
        h12_y0sum.columns=['species not found']
        h12_y1mean=y.mean(axis=0,level=['HUC12','COMID']).mean(axis=0,level='HUC12')
        h12_y1mean.columns=['share of species found']
        h12_ycount=h12_y1sum+h12_y0sum.values
        h12_ycount.columns=['potential species found']
        dfs=pd.concat([h12_y1sum,h12_y0sum,h12_y1mean,h12_ycount])
        cols=list(dfs.columns)
        c=2 # c* of olumns
        r=-(-len(cols)//c)
        tuplist=[(r,c,i+1) for i in range(len(cols))]
        #tuplist=[(len(cols),x+1,c) for x in range(len(cols)//c) for y in range(r)]
        
        df=pd.concat([h12_y1sum,h12_y0sum,h12_y1mean,h12_ycount])
        if huc_level is None:
            geo_df=self.hucBoundaryMerge(df)
        else:
            df_h,huc_level=self.hucAggregate(df,huc_level,collapse='mean')
            geo_df=self.hucBoundaryMerge(df_h,right_on=huc_level)
        self.geo_df=geo_df
        
        fig=plt.figure(dpi=600,figsize=[12,8])
        fig.suptitle('dependent variable')
        for i in range(len(cols)):
            col=cols[i]
            self.map_plot(geo_df,col,subplot_tup=tuplist[i],fig=fig)
        if huc_level:
            name=f'y01_{huc_level}.png'
        else:
            name='y01'+'.png'
        
        fig.savefig(Helper().getname(os.path.join(self.print_dir,name)))
        fig.show() 
        
    def plot_confusion_01predict(self,rebuild=0,fit_scorer=None,drop_zzz=True,wt=None,huc_level=None,normal_color=True):
        if fit_scorer is None:
            fit_scorer=self.fit_scorer
        coef_df,scor_df,y,yhat=self.cv_pr.get_coef_stack(rebuild=rebuild,drop_zzz=drop_zzz,return_y_yhat=True)
        if not wt is None:
            wt_df=self.cv_pr.build_wt_comid_feature_importance(rebuild=rebuild,return_weights=True)
            assert False,'not developed'

        yhat_a,y_a=yhat.align(y,axis=0)
        y_vals=y_a.values
        diff=yhat_a.subtract(y_vals,axis=0).astype(np.float16)*-1 # y-yhat


        pos=diff[y_vals==1]
        neg=diff[y_vals==0].abs()
        
        tp=(1-pos).mean(axis=1)
        tn=(1-neg).mean(axis=1)
        fp=neg.mean(axis=1) #dif is neg for an fp, so reverse it to signal fp
        fn=pos.mean(axis=1)
        
        
        
        confu_list=[tp,fp,fn,tn]
        confu_varnames=['true positive','false positive','false negative','true negative']
        for i in range(len(confu_list)):confu_list[i].rename(confu_varnames[i],inplace=True)
        
        self.confu_list=confu_list
        #confu_list=[self.swap_index_by_level(confu_list[i],'var',confu_varnames[i],axis=1) for i in range(4)]                 
        
        if not wt is None:
            assert False,'not developed'
        if huc_level is None:
            huc_level='HUC12' 
            confu_list=[ser.mean(axis=0,level=huc_level) for ser in confu_list] #ser for series
        else:
            #confu_df,huc_level=self.hucAggregate(confu_df,huc_level,collapse='mean')    
            confu_list,huc_levels=zip(*[self.hucAggregate(df,huc_level,collapse='mean') for df in confu_list])  
            huc_level=huc_levels[0]
        """   
        self.logger.info(f'starting confu_list join')
        confu_df=confu_list[0].join(confu_list[1:])
        self.logger.info(f'completed join')
        """
        
        geo_dfs=[self.hucBoundaryMerge(df,right_on=huc_level) for df in confu_list]
        self.geo_dfs=geo_dfs
        df_col_vars=confu_varnames #
        c=2 #  of columns
        r=2
        tuplist=[(r,c,i+1) for i in range(len(df_col_vars))]
        
        fig=plt.figure(dpi=600,figsize=[12,8])
        fig.suptitle(f'all species normalized confusion matrix by {huc_level}')
        for i in range(len(df_col_vars)):
            col=df_col_vars[i]
            geo_df=geo_dfs[i]
            self.logger.info(f'adding plot {i+1} of {len(df_col_vars)}')
            if normal_color:
                plotkwargs={'vmin':0,'vmax':1}
            else:
                plotkwargs={}
            self.map_plot(geo_df,col,subplot_tup=tuplist[i],fig=fig,plotkwargs=plotkwargs)

        name=f'confusion_matrix_map_{huc_level}.png'
        
        fig.savefig(Helper().getname(os.path.join(self.print_dir,name)))
        fig.show() 
                     
        
                     
    def swap_index_by_level(self,df,level,new,axis=0):
        if axis==0:
            idx=df.index
        elif axis==1:
            idx=df.columns
        else:
            assert False,'no other option'
        level_names=idx.names
        level_pos=level_names.index(level)
        assert type(idx) is pd.MultiIndex, f'expecting df to have multiindex index. but type(idx):{type(idx)}'
        new_idx=[]
        for tup in idx:
            not_tup=list(tup)
            not_tup[level_pos]=new
            new_idx.append(tuple(not_tup))
        new_midx=pd.MultiIndex.from_tuples(new_idx,names=level_names)
        if axis==0:
            df.index=new_midx
        elif axis==1:
            df.columns=new_midx
        else: 
            assert False,'no other option'
        return df    
        
        
        
    def hucAggregate(self,df,huc_level,collapse='mean'):
        if type(huc_level) is int:
            if huc_level<10:
                huc_name='HUC0'+str(huc_level)
            else:
                huc_name='HUC'+str(huc_level)
            huc_name='HUC'+str(int)
            huc_digs=str(huc_level)
        elif huc_level[:3].lower()=='huc':
            huc_digs=huc_level[3:]
            huc_name=huc_level
        else:
            huc_digs=huc_level
            huc_name='HUC'+str(huc_level)
        if huc_digs[0]==0:
            huc_digs=huc_digs[1:]
        huc_dig_int=int(huc_digs)
        idx1=df.index
        if not type(idx1) is pd.MultiIndex:
            idx1=pd.MultiIndex.from_tuples([(idx,) for idx in idx1],names=['HUC12']) # make into a tuple, so iterates like a multiindex
            huc_pos=0
        else:
            huc_pos=None
            for pos,name in enumerate(idx1.names):
                if re.search('huc',name.lower()):
                    huc_pos=pos
                    break
            assert not huc_pos is None, f'huc_pos is None!, idx1.index.names:{idx1.index.names}'
        
        idx2=[(idx1[i][huc_pos][:huc_dig_int],*idx1[i]) for i in range(len(idx1))] # idx1 is a list of tuples 
        ##'species','estimator','HUC12','COMID', so [-2]  is 'HUC12'
        #idx2=[(idx_add,*idx1[i]) for i in range(len(idx1))] # prepend idx_add
        
        
        expanded_midx=pd.MultiIndex.from_tuples(idx2,names=(huc_name,*idx1.names))
        
        df.index=expanded_midx
        if not collapse is None:
            if collapse=='mean':
                return df.mean(axis=0,level=huc_name),huc_name # mean across huc
            else: assert False, 'not developed'
        return df,huc_name
    
    def map_plot(self,gdf,col,add_huc_geo=None ,fig=None,ax=None,subplot_tup=(1,1,1),title=None,plotkwargs={}):
        if ax is None:
            if fig is None:
                savefig=1
                fig=plt.figure(dpi=300,figsize=[10,8])
            else:
                savefig=0
             
            ax=fig.add_subplot(*subplot_tup)
        if title is None:
            title=col
        ax.set_title(title)
        self.add_huc2_conus(ax,huc2_select=None)
        if not add_huc_geo is None:
            assert type(add_huc_geo) is str,f'expecting string like HUC8, but got: {add_huc_geo}'
            gdf=self.hucBoundaryMerge(gdf,right_on=add_huc_geo)
        self.gdf=gdf
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        gdf.plot(column=col,ax=ax,cax=cax,legend=True,**plotkwargs)#,legend_kwds={'orientation':'vertical'})
        #collection=self.plot_collection(ax,gdf.geometry,values=gdf[col],colormap='cool')
        
        if savefig:fig.savefig(Helper().getname(os.path.join(self.print_dir,title+'.png')))

    def plot_collection(self,ax, geoms, values=None, colormap='Set1',  facecolor=None, edgecolor=None,
                            alpha=.9, linewidth=1.0, **kwargs):
        patches = []

        for multipoly in geoms:
            for poly in multipoly:

                a = np.asarray(poly.exterior)
                if poly.has_z:
                    poly = shapely.geometry.Polygon(zip(*poly.exterior.xy))

                patches.append(Polygon(a))

        patches = PatchCollection(patches, facecolor=facecolor, linewidth=linewidth, edgecolor=edgecolor, alpha=alpha, **kwargs)

        if values is not None:
            patches.set_array(values)
            patches.set_cmap(colormap)

        ax.add_collection(patches, autolim=True)
        ax.autoscale_view()
        return patches
    
 
    def top_features_by_species_cluster(
        self,top_clusts=5,row_norm=True,clusterer='AgglomerativeClustering',n_clusters=5,est=None,
        huc_level=None,split=None,top_n=10,rebuild=0,zzzno_fish=False,
        filter_vars=True,spec_wt=None,fit_scorer=None,scale_by_X=False,
        presence_filter=True,wt_type='fitscor_diffscor',cv_collapse='split',):
        
        coef_df,scor_df=self.cv_pr.get_coef_stack(
            rebuild=rebuild,drop_zzz=not zzzno_fish,drop_nocoef_scors=True,row_norm=row_norm)
        if re.search('fit',wt_type):
            scor_wt=True
        else:
            scor_wt=False
        if fit_scorer is None:
            fit_scorer=self.fit_scorer
        self.skc=SkCluster(fit_scorer=fit_scorer)
        spec_clust_dict=self.skc.clusterSpecsByCoefs(
            coef_df,scor_df,clusterer=clusterer,
            n_clusters=n_clusters,zzzno_fish=zzzno_fish,
            cv_collapse=cv_collapse,scor_wt=scor_wt,
            est=est,fit_scorer=fit_scorer
            )
        self.spec_clust_dict=spec_clust_dict
        #assert False, 'debug'
        size_sorted_clusts=[key for _,key in sorted([(len(val),key) for key,val in spec_clust_dict.items()])[::-1]]
        spec_listlist=[]
        for clustkey in size_sorted_clusts[:top_clusts]:
            spec_listlist.append(spec_clust_dict[clustkey])
        for spec_list in spec_listlist:
            self.plot_top_features(
                huc_level=huc_level,split=split,top_n=top_n,rebuild=rebuild,
                zzzno_fish=zzzno_fish,filter_vars=filter_vars,spec_wt=spec_wt,
                fit_scorer=fit_scorer,scale_by_X=scale_by_X,presence_filter=presence_filter,
                wt_type=wt_type,cv_collapse=cv_collapse,spec_list=spec_list,row_norm=row_norm)
        
    
    def plot_top_features(
        self,huc_level=None,split=None,top_n=10,rebuild=0,zzzno_fish=False,
        filter_vars=True,spec_wt=None,fit_scorer=None,scale_by_X=False,
        presence_filter=False,wt_type='fitscor_diffscor',cv_collapse=False,
        spec_list=None,vote=False,row_norm=False):
        
        """
        coef_df,scor_df,y,yhat=self.cv_pr.get_coef_stack(
            rebuild=rebuild,drop_zzz=not zzzno_fish,return_y_yhat=True,
            drop_nocoef_scors=True)
        """
        
        if fit_scorer is None:
            fit_scorer=self.fit_scorer
        if presence_filter:
            title=f'top {top_n} presence-hi absence-lo features'
        else:
            title=f'top {top_n} hilo features'
        if scale_by_X:
            title+='_Xscaled'
        if row_norm:
            title+='_row-norm'
        if zzzno_fish:
            title+='_zzzno-fish'
        if filter_vars:
            title+='_var-filter'
        if spec_wt=='even':
            title+='_even-wt-spec'
        if type(wt_type) is str:
            title+=f'_{wt_type}'
        if huc_level:
            title+=f'_{huc_level}'
        if cv_collapse:
            title+="_cv-mean"
        if vote:
            title+='_var-vote'
        else:
            title+='_var-mean'
        if not spec_list is None:
            sp_hash=joblib.hash(spec_list)
            self.cv_pr.getsave_postfit_db_dict('spec_list',{sp_hash:spec_list})
            title+=f'sp_count{len(spec_list)}-hash{sp_hash}'
            print(f'starting plot for {len(spec_list)} species. spec_list:{spec_list}, with hash:{sp_hash}')
      
            
            
        title+='_'+fit_scorer
        
        wtd_coef_dfs=self.cv_pr.build_wt_comid_feature_importance(
            presence_filter=presence_filter,rebuild=rebuild,zzzno_fish=zzzno_fish,
            spec_wt=spec_wt,fit_scorer=fit_scorer,scale_by_X=scale_by_X,
            wt_type=wt_type,cv_collapse=cv_collapse,spec_list=spec_list,row_norm=row_norm)
        self.wtd_coef_dfs=wtd_coef_dfs
        if type(wtd_coef_dfs) is pd.DataFrame:
            wtd_coef_dfs=[wtd_coef_dfs]
        
        for i,wtd_coef_df_ in enumerate(wtd_coef_dfs):
            
            self.logger.info(f'pi_map dropping cols from wtd_coef_df i:{i}')
            #wtd_coef_df_.astype(np.float32,copy=False)
            cols=wtd_coef_df_.columns
            if filter_vars:
                drop_vars=[
                    'tmean','tmax','msst','mwst','precip',
                    'slope','wa','elev','mast','tmin','Al2O3','slp']
                for col in cols:
                    for varstr in drop_vars:
                        if re.search(varstr,col.lower()):
                            try:
                                wtd_coef_df_.drop(col,axis=1,inplace=True)
                                self.logger.info(f'dropped col: {col}')
                            except:
                                self.logger.info(f'failed trying to drop col:{col} matching varstr:{varstr} and current cols:{list(wtd_coef_df_.columns)}')
                            
                            break #stop searching
        if split is None:
            
            self.plot_hilo_coefs(wtd_coef_dfs,top_n,title,huc_level=huc_level,vote=vote)
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
        
    def plot_hilo_coefs(self,wtd_coef_dfs,top_n,title,huc_level=None,vote=False):
        cols_sorted_list=[];big_top_2n=[]
        selector=[0,-1]
        select_cols_list=[]
        lo_hi_names=['lowest_coef','highest_coef']
        for df_idx in selector:
            wtd_coef_df=wtd_coef_dfs[df_idx]
            #if not huc_level is None:
            #    wtd_coef_df,_=self.hucAggregate(wtd_coef_df,huc_level)
        
            if vote:     
                #the variable names    
                cols=wtd_coef_df.columns
                #sort acros cols, i.e., each row
                coef_sort_idx=np.argsort(wtd_coef_df,axis=1) 

                select_cols=coef_sort_idx.apply(
                    lambda x:cols[x.iloc[df_idx]],axis=1,) #df_idx will choose the first or last from each sorted list

                n_select_cols=select_cols.value_counts(ascending=False).index.tolist()[:top_n]
                big_top_2n.append(n_select_cols)
                #big_top_2n=tuple(select_cols_list)

            else:
                big_mean=wtd_coef_df.mean(axis=0)#mean aross all hucs
                #sort_ord=np.argsort(big_mean)
                #cols_sorted_list.append(wtd_coef_df.columns.to_list()[sort_ord])
                cols_sorted=(list(big_mean.sort_values().index))#ascending
                if df_idx==0:
                    top_slice=slice(0,top_n)
                else:
                    top_slice=slice(-top_n,None)
                big_top_2n.append(cols_sorted[top_slice],
                        ) #-1 could be last item or 1st if len=1
        big_top_2n=tuple(big_top_2n)
        top_neg_coef_df=wtd_coef_dfs[0].loc[:,big_top_2n[0]]
        top_pos_coef_df=wtd_coef_dfs[-1].loc[:,big_top_2n[1]]
        #self.big_top_wtd_coef_df=big_top_wtd_coef_df
        pos_sort_idx=np.argsort(top_pos_coef_df,axis=1) #ascending
        neg_sort_idx=np.argsort(top_neg_coef_df,axis=1)#.iloc[:,::-1] # ascending
        #select the best and worst columns
        big_top_cols_pos=pos_sort_idx.apply(
            lambda x: big_top_2n[1][x.iloc[-1]],axis=1)
        big_top_cols_neg=neg_sort_idx.apply(
            lambda x: big_top_2n[0][x.iloc[0]],axis=1)
        colname='top_predictive_variable'
        big_top_cols_pos=big_top_cols_pos.rename(colname)
        big_top_cols_neg=big_top_cols_neg.rename(colname)
        
        self.big_top_cols_pos=big_top_cols_pos
        self.big_top_cols_neg=big_top_cols_neg

        self.logger.info('starting boundary merge pos')
        
        geo_pos_cols=self.hucBoundaryMerge(big_top_cols_pos)


        fig=plt.figure(dpi=300,figsize=[10,14])
        fig.suptitle(title)
        ax=fig.add_subplot(2,1,1)
        ax.set_title('top positive features')
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.1)
        geo_pos_cols.plot(column=colname,ax=ax,cmap='tab20c',legend=True)#,legend_kwds={'orientation':'vertical'})
        self.add_huc2_conus(ax)

        self.logger.info('starting boundary merge neg')
        geo_neg_cols=self.hucBoundaryMerge(big_top_cols_neg)

        ax=fig.add_subplot(2,1,2)
        ax.set_title('top negative features')
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.1)
        geo_neg_cols.plot(column=colname,ax=ax,cmap='tab20c',legend=True)#,legend_kwds={'orientation':'vertical'})
        self.add_huc2_conus(ax)
        
        figname=Helper().getname(os.path.join(
            self.print_dir,f'huc12_top{top_n}_features_{title}.png'))
        self.logger.info(f'saving to {figname}')
        fig.savefig(figname)
    
        
    def add_huc2_conus(self,ax,huc2_select=None):
        try: huc2=self.boundary_dict['huc02']
        except: 
            huc2=self.getHucBoundary('huc02')
        if huc2_select is None:
            huc2_conus=huc2.loc[huc2.loc[:,'huc2'].astype('int')<19,'geometry']
        else:
            huc2_conus=huc2.loc[huc2.loc[huc2_select,'huc2'].astype('int')<19,'geometry']
        huc2_conus.boundary.plot(linewidth=1,color=None,edgecolor='k',ax=ax)
   
        
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

"""   
class XMapper(Mapper,PiscesPredictDataTool):
    def __init__(self,species,estimator=''):
        super().__init__()
        #myLogger.__init__(self,name='Mapper.log')
        #Mapper.__init__(self)
        #PiscesPredictDataTool.__init__(self)
        #NHD data downloaded from https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/National/HighResolution/GDB/NHD_H_National_GDB.zip
"""
class mp_mapper_runner(Process):
    def __init__(self,q,mapper,kwargs={}):
        super().__init__()
        self.q=q
        self.mapper=mapper
        self.kwargs=kwargs
        
    def run(self):
        while not self.q.empty():
            try:
                spec=self.q.get_nowait()
                print(f'mapping {spec}')
                self.mapper.plotSpeciesPredict(spec,**self.kwargs)
            except:
                print(f'{spec} error {format_exc()}')
        return
        

if __name__=="__main__":
    try:
        mpr=Mapper(mp=True)
        specs=PiscesPredictDataTool(cv_run=False).returnspecieslist()
        shuffle(specs)
        q=Queue()
        for spec in specs:q.put(spec)
        proc_count=7
        plot_kwargs=dict(huc_level=4,include_absent=False,save_check=True,plot_train=False,main_plot='cv_probability')
        proc_list=[mp_mapper_runner(q,mpr,kwargs=plot_kwargs) for _ in range(proc_count)]
        for proc in proc_list:proc.start()
        for proc in proc_list:proc.join()
                        
    except:
        print(format_exc())    
            
        

