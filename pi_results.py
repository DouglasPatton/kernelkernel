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
from datagen import dataGenerator 
from pisces_params import PiSetup,MonteSetup
from pi_db_tool import DBTool

def PiResults(DBTool,myLogger):
    def __init__(self,):
        func_name=f'{sys._getframe().f_code.co_name}'
        myLogger.__init__(self,name=f'{func_name}.log')
        self.logger.info(f'starting {func_name} logger')
        DBTool.__init__(self)
        self.resultsDBdict=self.resultsDBdict()
        
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
                    spec_est_scr_dict[species][key[5:]]=model[key]
                
    