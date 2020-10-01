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
from pi_data_viz import DataPlotter
from helpers import Helper
import pickle

class PiResults(DBTool,DataPlotter,myLogger):
    '''
    from qcluster.runnode: 
        model_dict={'model':SKToolInitializer(model_gen),'data_gen':data_gen,'model_gen':model_gen}
    '''
    def __init__(self,):
        func_name=f'PiResults'
        myLogger.__init__(self,name=f'{func_name}.log')
        self.logger.info(f'starting {func_name} logger')
        DBTool.__init__(self)
        DataPlotter.__init__(self)
        self.resultsDBdict=self.resultsDBdict()
        self.printdir=os.path.join(os.getcwd(),'print')
        if not os.path.exists(self.printdir):os.mkdir(self.printdir)
        self.scor_est_spec_dict={}
        self.sk_est_dict=sk_estimator().get_est_dict() 
        self.scorer_list=list(SKToolInitializer(None).get_scorer_dict().keys())
        self.helper=Helper()
      
    def build_predict_est_spec_dict(self,):
        for hash_id,result_dict in self.resultsDBdict.items():
            pass
    
    def plot_species_estimator_predict(self):
        pass
    
    def build_scor_est_spec_dict(self,rebuild=0):
        savename=os.path.join('results','scor_est_spec_dict.pkl')
        
        if rebuild:
            scor_est_spec_dict={scorer:{est:{} for est in self.sk_est_dict.keys()} for scorer in self.scorer_list}
            for hash_id,result_dict in self.resultsDBdict.items():
                data_gen=result_dict['data_gen']
                species=data_gen['species']
                model_gen=result_dict['model_gen']
                est_name=model_gen['name']
                model=result_dict['model']
                if 'cv' in data_gen['data_split']:
                    model_keys=list(model.keys())
                    #test_result_keys=[key for key in model_keys if key[:5]=='test_']
                    test_result_keys=[f'test_{scorer}' for scorer in self.scorer_list]
                    for key in test_result_keys:
                        val=model[key]
                        scor_est_spec_dict[key[5:]][est_name][species]=val
            scor_est_spec_dict=self.drop_nan_species(scor_est_spec_dict)
            self.scor_est_spec_dict=scor_est_spec_dict
            self.save_dict(scor_est_spec_dict,filename=savename,bump=1,load=0)
        else:
            self.scor_est_spec_dict=self.save_dict(None,filename=savename,load=1)

    def drop_nan_species(self,sepd):
        drop_specs=[]
        for s,epd in sepd.items():
            for e,pd in epd.items():
                for p,scor in pd.items():
                    if any([np.isnan(scor_i) for scor_i in scor]):
                        drop_specs.append(p)
        for s,epd in list(sepd.items()):
            for e,pd in list(epd.items()):
                for p in list(pd.keys()):
                    if p in drop_specs:
                        sepd[s][e].pop(p)
        return sepd
        
        
    
    
    def save_dict(self,a_dict,filename='test.pkl',bump=1,load=0):
        if load:
            with open(filename,'rb') as f:
                dict_=pickle.load(f)
            return dict_
        if os.path.exists(filename) and bump:
            bumped_name=self.helper.getname(filename)
            with open(filename,'rb') as f:
                dict_=pickle.load(f)
            with open(bumped_name,'wb') as f:
                pickle.dump(dict_,f)
        with open(filename,'wb') as f:
            pickle.dump(a_dict,f)
            
                
                
    def plot_species_estimator_scores(self,):
        scorer_list=self.scorer_list
        scorer_count=len(scorer_list)
        est_list=list(self.sk_est_dict.keys())
        est_count=len(est_list)
        try: self.fit_sorted_species_dict,self.scor_est_spec_MLU
        except:self.build_mean_score_sort_spec_and_MLU()
        fig=plt.figure(dpi=300,figsize=[10,scorer_count*4])
        for s,scorer in list(enumerate(scorer_list)):
            sorted_species_list=self.fit_sorted_species_dict[scorer]
            est_spec_MLU=self.scor_est_spec_MLU[scorer]
            just_numbers=np.arange(len(sorted_species_list))
            ax=fig.add_subplot(scorer_count,1,s+1)
            ax.set_title(f'results for scorer:{scorer}')
            for e,est_name in list(enumerate(est_list))[:]:
                spec_MLU=est_spec_MLU[est_name]
                mean,lower,upper=zip(*[spec_MLU[spec] for spec in sorted_species_list]) # best across estimators, 
                mean_arr=np.array(mean)
                lower_arr=np.array(lower)
                upper_arr=np.array(upper)
                ax.margins(0)
                self.makePlotWithCI(
                        just_numbers,mean_arr,None,
                        ax,plottitle=est_name,color=e,
                        hatch=e,ls=e,lower=lower_arr,upper=upper_arr)
            ax.legend(loc=8,ncol=2)
            ax.set_xticks([])
        fig.show()
        self.fig=fig
        #figpath=self.helper.getname(os.path.join(self.printdir,f'test_scores_by_species_{scorer}.png'))
        #fig.savefig(figpath)
        
    def build_mean_score_sort_spec_and_MLU(self):
        try: scor_est_spec_dict=self.scor_est_spec_dict
        except: 
            self.build_scor_est_spec_dict()
            scor_est_spec_dict=self.scor_est_spec_dict
        scorer_list=self.scorer_list
        scorer_count=len(scorer_list)
        est_list=list(self.sk_est_dict.keys())
        est_count=len(est_list)
        species_list=list(set([spec for s,est in self.scor_est_spec_dict.items() for est,spec_dict in est.items() for spec in spec_dict.keys()]))
        
        self.logger.info(f'species_list: {species_list}')
        
        scor_spec_est_mean_dict={scorer:{spec:{est:None for est in est_list} for spec in species_list} for scorer in scorer_list}
        scor_est_spec_MLU={scorer:{est:{spec:None for spec in species_list} for est in est_list} for scorer in scorer_list}
        # create a separate graph for each scorer. with each estimator on each graph
        
        
        for s,scorer in enumerate(scorer_list):
            est_spec_dict=self.scor_est_spec_dict[scorer]

            
            for e,est_name in enumerate(est_list):
                try:
                    spec_dict=est_spec_dict[est_name] # this may trigger the exception
                    score_arr_list=[] # ordered by species_list
                    for spec in species_list:
                        if spec in spec_dict:
                            score_arr_list.append(np.array(spec_dict[spec]))
                        else:
                            score_arr_list.append(np.array([np.nan]))
                    mean_scores=[np.nanmean(scores) for scores in score_arr_list] #ordered by species_list
                    
                    for i in range(len(species_list)):
                        scor_spec_est_mean_dict[scorer][species_list[i]][est_name]=mean_scores[i]
                    sorted_score_arr_list=[np.sort(arr) for arr in score_arr_list] # ordered by species_list
                    len_list=[arr.shape[0]-1 for arr in sorted_score_arr_list] 
                    l_idx,u_idx=zip(*[(int(l*0.025),int(-(-(l*0.975)//1))) for l in len_list]) 
                    #self.logger.warning(f'len_list:{len_list},l_idx:{l_idx}, u_idx:{u_idx}')
                    lower,upper=zip(*[(arr[l_idx[i]],arr[u_idx[i]]) for i,arr in enumerate(sorted_score_arr_list)])
                    for i in range(len(species_list)):
                        scor_est_spec_MLU[scorer][est_name][species_list[i]]=(mean_scores[i],lower[i],upper[i])
                    #    MLU mean lower upper
                    
                except:
                    self.logger.exception('')
        scor_spec_bestesttup_dict=self.build_best_est_dict(scor_spec_est_mean_dict) 
        #    scorer:species:(val,est) where val is lowest across ests for scorer and species
        fit_sorted_species_dict={}
        for s,scorer in list(enumerate(scorer_list)):
            est_spec_MLU=scor_est_spec_MLU[scorer]
            spec_best_est_dict_s=scor_spec_bestesttup_dict[scorer]
            best_scor_est_tups=[spec_best_est_dict_s[spec] for spec in species_list]#ordering by species_list
            
            bestscors=[tup[0] for tup in best_scor_est_tups]
            species_scorsort_idx=list(np.argsort(np.array(bestscors))) #argsort to handle np.nan in the list
            best_cvmean_scors=[best_scor_est_tups[i][1] for i in species_scorsort_idx]
            
            sorted_species_list=[species_list[i] for i in species_scorsort_idx]
            #plt.scatter(sorted_species_list,best_cvmean_scors)
            #plt.show()
            fit_sorted_species_dict[scorer]=sorted_species_list
        self.fit_sorted_species_dict=fit_sorted_species_dict
        self.scor_est_spec_MLU=scor_est_spec_MLU
            
    def build_best_est_dict(self,scor_spec_est_mean_dict,):
        scor_spec_bestesttup_dict={}
        for scorer,spec_dict in scor_spec_est_mean_dict.items():
            spec_best_est_dict={}
            for spec,est_dict in spec_dict.items():
                ests,scors=zip(*[(est,scor) for est,scor in est_dict.items()])
                maxscor=np.nanmax(np.array([scors]))
                max_idx=scors.index(maxscor)
                spec_best_est_dict[spec]=(scors[max_idx],ests[max_idx]) # (val,name) for sorting later
            scor_spec_bestesttup_dict[scorer]=spec_best_est_dict
        return scor_spec_bestesttup_dict
                
