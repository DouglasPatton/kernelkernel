import pickle
import os,sys,psutil
import re
from time import strftime,sleep
import datetime
import traceback
import shutil
from random import randint,seed,shuffle
import logging
from helpers import Helper
import multiprocessing as mp
from multiprocessing.managers import BaseManager
from queue import Queue
import numpy as np
from sqlitedict import SqliteDict
from  sk_tool import SKToolInitializer
from sk_estimators import sk_estimator
from datagen import dataGenerator 
from pisces_params import PiSetup,MonteSetup
from mylogger import myLogger
from pi_db_tool import DBTool
import matplotlib.pyplot as plt

class PiResults(DBTool,myLogger):
    def __init__(self,):
        func_name=f'{sys._getframe().f_code.co_name}'
        myLogger.__init__(self,name=f'{func_name}.log')
        self.logger.info(f'starting {func_name} logger')
        DBTool.__init__(self)
        self.resultsDBdict=self.resultsDBdict()
        self.spec_est_scr_dict={}
        self.sk_est_dict=sk_estimator().get_est_dict() 
        self.scorer_list=SKToolInitializer.get_scorer_list(None) # don't need to initialize 
        
    def build_species_estimator_score_dict(self,):
        spec_est_scr_dict={}
        for hash_id,result_dict in self.resultsDBdict.items():
            data_gen=result_dict['data_gen']
            species=data_gen[species]
            model_gen=result_dict['model_gen']
            est_name=model_gen['name']
            model=result_dict['model']
            if not species in spec_est_scr_dict:
                spec_est_scr_dict[species]={}
            if 'cv' in data_gen['data_split']:
                model_keys=list(model.keys())
                test_result_keys=[key for key in model_keys if key[:5]=='test_']
                for key in test_result_keys:
                    spec_est_scr_dict[species][key]=model[key]
        self.spec_est_scr_dict=spec_est_scr_dict
                    
    def plot_species_estimator_scores(self):
        try: self.spec_est_scr_dict
        except: self.build_species_estimator_score_dict()
        fig=plt.figure(dpi=600,figsize=[10,6])
        plt.xticks(rotation=65)
        scorer_list=self.scorer_list
        scorer_count=len(scorer_list)
        est_list=list(self.sk_est_dict.keys())
        est_count=len(est_list)
        for s,scorer in enumerate(scorer_list):
            ax=fig.subplot(s+1,1,est_count)
            ax.set_title(f'results for scorer:{scorer}')
            for e,est_name in enumerate(est_list):
                
            

        
        ax=fig.subplot()
        ax.set_title('f1_')#Fixed Effects Estimates for Water Clarity by Distance from Shore Band')
        ax.set_xlabel('Distance from Shore Bands (not to scale)')
        ax.set_ylabel('Partial Derivatives of Sale Price by Water Clarity ($/m)')
        
        for p in [0,1]:
            bigX_dist_avg_df_p=bigX_dist_avg_df_list[p]
            #print('bigX_dist_avg_df_p',bigX_dist_avg_df_p)
            effect=bigX_dist_avg_df_p[f'marginal_p{p}'].to_numpy(dtype=np.float64)
            lower=bigX_dist_avg_df_p[f'lower95_marginal_p{p}'].to_numpy(dtype=np.float64)
            upper=bigX_dist_avg_df_p[f'upper95_marginal_p{p}'].to_numpy(dtype=np.float64)
            #print('effect',effect)
            #print('lower',lower)
            #print('upper',upper)
            self.makePlotWithCI(effect_name,effect,None,ax,**self.plot_dict_list[p],lower=lower,upper=upper)
        ax.legend(loc=1)
        ax.margins(0)
        #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('$%.f'))
        figpath=self.helper.getname(os.path.join(self.printdir,'wq_effects_graph.png'))
        fig.savefig(figpath)
                
    