from sqlitedict import SqliteDict
from shutil import copy
import sys,os,logging
from mylogger import myLogger
import zlib, pickle, sqlite3
from time import sleep
from traceback import format_exc

class DBTool():
    def __init__(self,cv_run=None):
        assert not cv_run is None,f'expecting True or False for cv_run:{cv_run}'
        func_name='DBTool'
        #super.__init__(name=f'{func_name}.log')
       
        self.logger=logging.getLogger()
        #
        results_dir_name='results'
            
        if cv_run:
            results_dir_name+='_cv'
        resultsdir=os.path.join(os.getcwd(), results_dir_name)

        self.resultsdir=resultsdir
        if not os.path.exists(resultsdir):
            os.mkdir(resultsdir)
        self.errordir=os.path.join(resultsdir,'error')
        if not os.path.exists(self.errordir):
            os.mkdir(self.errordir)
        self.resultsDBdictpath=os.path.join(resultsdir,'resultsDB.sqlite')
        self.noveltyFilterResultsDBdictpath=os.path.join(resultsdir,'noveltyFilterResultsDB.sqlite')
        self.metadataDBdictpath=os.path.join(resultsdir,'metadataDB.sqlite')
        self.genDBdictpath=os.path.join(resultsdir,'genDB.sqlite')
        self.noveltyFilterenDBdictpath=os.path.join(resultsdir,'noveltyFilterGenDB.sqlite')
        self.predictDBdictpath=os.path.join(resultsdir,'predictDB.sqlite')
        self.XpredictDBdictpath=os.path.join(resultsdir,'XpredictDB.sqlite')
        self.Xpredictspecies_dir=os.path.join(resultsdir,'species_predict')
        if not os.path.exists(self.Xpredictspecies_dir): 
            os.mkdir(self.Xpredictspecies_dir)
        self.NoveltyFilterDBdictpath=os.path.join(resultsdir,'NoveltyFilterdDB.sqlite')
        self.NoveltyFilterspecies_dir=os.path.join(resultsdir,'species_novelty')
        if not os.path.exists(self.NoveltyFilterspecies_dir): 
            os.mkdir(self.NoveltyFilterspecies_dir)
        self.postfitDBdictpath=os.path.join(resultsdir,'postfitDB.sqlite')
        self.fitfailDBdictpath=os.path.join(resultsdir,'fitfailDB.sqlite')
        
        self.pidataDBdictpath=os.path.join(os.getcwd(),'data_tool','pidataDB.sqlite')
        
        
        #self.resultsDBdict=lambda:SqliteDict(filename=self.resultsDBdictpath,tablename='results') # contains sk_tool for each hash_id
        #self.genDBdict=lambda:SqliteDict(filename=self.resultsDBdictpath,tablename='gen')# gen for generate. contains {'model_gen':model_gen,'data_gen':data_gen} for each hash_id
        #self.predictDBdict=lambda name:SqliteDict(filename=self.predictDBdictpath,tablename=name)
    def re_encode(self,path):
        assert os.path.exists(path),f'{path} does not exist'
        bpath=path+'_backup'
        os.rename(path,bpath)
        #os.remove(path)
        tablenames=SqliteDict.get_tablenames(bpath)
        for name in tablenames:
            self.logger.info(f'starting tablename:{name}')
            with SqliteDict(filename=bpath,tablename=name,) as olddict:
                with SqliteDict(filename=path,tablename=name,encode=self.my_encode, decode=self.my_decode) as newdict:
                    keys=list(olddict.keys())
                    kcount=len(keys)
                    self.logger.info(f'for tablename:{name}, keys: {keys}')
                    for k,key in enumerate(keys):
                        if (k+1)%100==0:
                            print(f'{k}/{kcount}.',end='')
                        val=olddict[key]
                        newdict[key]=val
                        newdict.commit()
    
    def get_tablenames(self,path):
        return SqliteDict.get_tablenames(path)
    
    def anyNameDB(self,dbname,tablename='data',folder=None):
        
        name=dbname+'.sqlite'
        if folder is None:
            path=os.path.join(self.resultsdir,name)
        else:
            path=os.path.join(folder,name)
        return SqliteDict(
            filename=path,tablename=tablename,
            encode=self.my_encode,decode=self.my_decode)
    @staticmethod
    def my_encode(obj,chunk_size=None):
        try:
            if chunk_size is None:
                return sqlite3.Binary(zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL),level=9))
            else:
                assert type(chunk_size) in [int, float], f'expecting int or float for chunk_size, but got: {type(chunk_size)}'
                chunk_list=[]
                big_pickle=pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
                pickle_len=len(big_pickle)
                chunk_count=int(-(-pickle_len//chunk_size))
                for i in range(chunk_count):
                    chunk_list.append(zlib.compress(big_pickle[int(chunk_size*i):int(chunk_size*(i+1))],level=5))
                    #chunk_list.append(big_pickle[int(chunk_size*i):int(chunk_size*(i+1))])

                return chunk_list
                    
                    
        
        except:
            print(format_exc())
            assert False,'encode error'
        
        
    @staticmethod    
    def my_decode(obj):
        try:
            if type(obj) is list:
                chunk_list=[]
                for ch in obj:
                    full_ch=zlib.decompress(bytes(ch))
                    #full_ch=bytes(ch)
                    #print('type(full_ch): ',type(full_ch))
                    chunk_list.append(full_ch)
                pickled_string=b''.join(chunk_list)
                return pickle.loads(pickled_string)
                #return pickle.loads(''.join([zlib.decompress(bytes(ch)) for ch in obj]))

            return pickle.loads(zlib.decompress(bytes(obj)))
        except:
            print(format_exc())
            assert False,'encode error'
    #mydict = SqliteDict('./my_db.sqlite', encode=self.my_encode, decode=self.my_decode)
    
    def XpredictSpeciesResults(self,spec,hash_id):
        name=f'XpredictSpeciesResults-{spec}'

        return self.anyNameDB(name,tablename=hash_id,folder=self.Xpredictspecies_dir)
    
    def NoveltyFilterSpeciesResults(self,spec,hash_id):
        name=f'NoveltyFilterSpeciesResults-{spec}'

        return self.anyNameDB(name,tablename=hash_id,folder=self.NoveltyFilterspecies_dir)
    
    def XpredictHashIDComidHashResultsDB(self,hash_id=None):
        name='XpredictHashIDComidHashResults' #note lowercase p!
        if hash_id is None:
            path=os.path.join(self.resultsdir,name+'.sqlite')
            if not os.path.exists(path):
                return []
            try:
                return SqliteDict.get_tablenames(path)
            except OSError:
                return []
            except:
                assert False,f'unexpected error, hash_id:{hash_id}'
        return self.anyNameDB(name,tablename=hash_id,folder=self.resultsdir)

    def NoveltyFilterHashIDComidHashResultsDB(self,hash_id=None):
        name='NoveltyFilterHashIDComidHashResults' 
        if hash_id is None:
            path=os.path.join(self.resultsdir,name+'.sqlite')
            if not os.path.exists(path):
                return []
            try:
                return SqliteDict.get_tablenames(path)
            except OSError:
                return []
            except:
                assert False,f'unexpected error, hash_id:{hash_id}'
        return self.anyNameDB(name,tablename=hash_id,folder=self.resultsdir)
    
    
    def postFitDBdict(self,name):
        return SqliteDict(filename=self.postfitDBdictpath,tablename=name, encode=self.my_encode, decode=self.my_decode)
    
    def fitfailDBdict(self):
        return SqliteDict(filename=self.fitfailDBdictpath,tablename='fitfail', encode=self.my_encode, decode=self.my_decode) 
    
    def resultsDBdict(self):
        return SqliteDict(filename=self.resultsDBdictpath,tablename='results', encode=self.my_encode, decode=self.my_decode)
    
    def noveltyFilterResultsDBdict(self):
        return SqliteDict(filename=self.noveltyFilterResultsDBdictpath,tablename='results', encode=self.my_encode, decode=self.my_decode)
    
    def metadataDBdict(self):
        return SqliteDict(filename=self.metadataDBdictpath,tablename='metadata', encode=self.my_encode, decode=self.my_decode)
    
    def resultsDBdict_backup(self):
        return SqliteDict(filename='results/resultsDB_backup.sqlite',tablename='results', encode=self.my_encode, decode=self.my_decode)
    
    def pidataDBdict(self,name='species01'):
        return SqliteDict(filename=self.pidataDBdictpath,tablename=name, encode=self.my_encode, decode=self.my_decode)
    
    def noveltyFilterGenDBdict(self):
        return SqliteDict(filename=self.noveltyFilterGenDBdictpath,tablename='gen', encode=self.my_encode, decode=self.my_decode)
    
    def genDBdict(self):
        return SqliteDict(filename=self.genDBdictpath,tablename='gen', encode=self.my_encode, decode=self.my_decode)
    
    def predictDBdict(self,):
        return SqliteDict(filename=self.predictDBdictpath,tablename='predict01', encode=self.my_encode, decode=self.my_decode)
    
    def XpredictDBdict(self,):
        return SqliteDict(filename=self.XpredictDBdictpath,tablename='predict01', encode=self.my_encode, decode=self.my_decode)
    
    def NoveltyFilterDBdict(self,):
        return SqliteDict(filename=self.NoveltyFilterDBdictpath,tablename='novelty', encode=self.my_encode, decode=self.my_decode)
    
    def addToDBDict(self,save_list,db=None,gen=0,predict=0,pi_data=0,Xpredict=False,fast_add=False,is_novelty_filter=False):
        try:
            if type(save_list) is dict:
                save_list=[save_list]
            if not db is None:
                pass        
            elif gen:
                if is_novelty_filter:
                    db=self.noveltyFilterGenDBdict
                else:
                    db=self.genDBdict
            elif predict:
                assert not is_novelty_filter
                db=self.predictDBdict
            elif pi_data:
                assert not is_novelty_filter
                if type(pi_data) is str:
                    kwargs={'name':pi_data}
                else:
                    kwargs={}
                db=lambda: self.pidataDBdict(**kwargs)
            elif Xpredict:
                assert not is_novelty_filter
                db=lambda X:self.XpredictHashIDComidHashResultsDB(hash_id=X)
            else:
                if is_novelty_filter:
                    db=self.noveltyFilterResultsDBdict
                else:
                    db=self.resultsDBdict
            saved=False;tries=0
            while len(save_list)>0:
                try:
                    dict_i=save_list.pop()
                    if Xpredict:
                        for hash_id,c_hash_dict in dict_i.items():
                            with db(hash_id) as dbdict:
                                for c_hash,result in c_hash_dict.items():
                                    if c_hash in dbdict:
                                        self.logger.critical(f'c_hash:{c_hash} already in resultdict for hash_id:{hash_id}. old result: {dbdict[c_hash]}')
                                    dbdict[c_hash]=result
                                    dbdict.commit()
                        self.logger.info(f'add to dbdict success')
                    else:    
                        with db() as dbdict:
                            for key,val in dict_i.items():
                                if key in dbdict:
                                    if not gen:
                                        self.logger.warning(f'overwriting val:{dbdict[key]} for key:{key}')
                                        dbdict[key]=val
                                        dbdict.commit()
                                    else:
                                        pass
                                        #self.logger.debug(f'key:{key} already exists in gen table in db dict')
                                else:
                                    if fast_add:
                                        dbdict[key]=val
                                        continue
                                    else:
                                        tries2=0
                                        while True:
                                            try:
                                                self.logger.info(f'trying to add key-{key} to {db}')
                                                dbdict[key]=val
                                                dbdict.commit()
                                                self.logger.info(f'key-{key} added to {db}')
                                                break
                                            except:
                                                tries2+=1
                                                self.logger.exception(f'error adding to key:{key}, tries2:{tries2}')
                                                if tries2>3:
                                                    path=os.path.join(self.errordir,'results-'+key+'.pkl')
                                                    if not os.path.exists(path):
                                                        with open(path,'wb') as f:
                                                            pickle.dump(val,f)
                                                        self.logger.info(f'dumped to {path}')
                                                        try:
                                                            dbdict[key]=path
                                                            dbdict.commit()
                                                            self.logger.info(f'saved path to dbdict for key:{key}')
                                                        except:
                                                            self.logger.exception(f'could not save path to key:{key}, abandoning')

                                                    else:
                                                        self.logger.info(f'{path} exists, so ignoring')
                                                    break
                            if fast_add: dbdict.commit()
                                            
                except:
                    tries+=1
                    self.logger.exception(f'dbtool addtoDBDict error! tries:{tries}')
                    sleep(20)
                    if tries>4:
                        self.logger.warning(f'abandoning save_list:{save_list}')
                        break
                
            return  
        except:
            self.logger.exception(f'addToDBDict outer catch')
    
    def get_no_results_run_record_dict(self,ignore_failed=True):
        no_results_run_record_dict={}
        gen_dict=self.genDBdict()
        results_keys=dict.fromkeys(self.resultsDBdict().keys())
        if ignore_failed:
            fail_dict=self.fitfailDBdict()
        else:
            fail_dict={}
        for hash_id in gen_dict.keys():
            if not (hash_id in results_keys or hash_id in fail_dict):
                self.logger.info(f'no results or fails for gen_dict[hash_id]:{gen_dict[hash_id]}')
                no_results_run_record_dict[hash_id]=gen_dict[hash_id]
            else:self.logger.info(f'results or fails found for hash_id:{hash_id}')    
        return no_results_run_record_dict
                
    
    
    
    def purgeExtraGen(self):
        rdb=self.resultsDBdict()
        with self.genDBdict() as dbdict:
            for hash_id,run_record_dict in dbdict.items():
                if not hash_id in rdb:
                    self.logger.info(f'purging {hash_id} run_record_dict:{run_record_dict}')
                    del dbdict[hash_id]
            dbdict.commit()
            
            
    def del_est_from_results(self):
        rdb=self.resultsDBdict
        spec_xvar_dict={} # to store the xvars the first time a species comes up
        with rdb() as rdb_dict:
            for hash_id in list(rdb_dict.keys()):# generator to list so rdb_dict can be modified.
                modeldict=rdb_dict[hash_id]
                #data_gen=modeldict["data_gen"]
                #species=data_gen["species"]
                #est_name=modeldict["model_gen"]["name"]
                
                if type(modeldict['model']) is dict:
                    changed=False
                    for m_idx in range(len(modeldict['model']['estimator'])):
                        try:
                            del modeldict['model']['estimator'][m_idx].est
                            changed=True # if triggered once, will write back to db
                        except AttributeError:
                            self.logger.exception(f'no est attribute for hash_id:{hash_id}')
                        except:
                            self.logger.exception(f'halting bc unexpected error for modeldict:{modeldict}')
                            assert False,'halt'
                        
                    if changed:
                        rdb_dict[hash_id]=modeldict # db only updated if attribute added
                    
                    
                else:
                    assert False,'expecting cross_validate dict '

            rdb_dict.commit()
        return
            
        
        
    def add_x_vars_to_results(self):
        from datagen import dataGenerator

        rdb=self.resultsDBdict
        spec_xvar_dict={} # to store the xvars the first time a species comes up
        with rdb() as rdb_dict:
            for hash_id in list(rdb_dict.keys()):# generator to list so rdb_dict can be modified.
                modeldict=rdb_dict[hash_id]
                data_gen=modeldict["data_gen"]
                species=data_gen["species"]
                est_name=modeldict["model_gen"]["name"]
                
                if type(modeldict['model']) is dict:
                    try:
                        modeldict['model']['estimator'][0].x_vars
                        self.logger.info(f'{species}:{est_name} already has x_vars')
                    except AttributeError: # add the attribute
                        try:
                            x_vars=spec_xvar_dict[species]
                        except KeyError:
                            self.logger.info(f'key error for {species}:{est_name}, so calling dataGenerator')
                            data=dataGenerator(data_gen)
                            x_vars=data.x_vars
                            spec_xvar_dict[species]=x_vars
                        except:
                            self.logger.exception(f'not a keyerror, unexpected error')
                            assert False,'halt'
                        self.logger.warning(f'adding x_vars for {species}:{est_name}')
                        for m in range(len(modeldict['model']['estimator'])): # cross_validate stores a list of the estimators
                            modeldict['model']['estimator'][m].x_vars=x_vars 
                            #    ^this should be the sktool object that contains the sk_estimator at sktool.model_
                        rdb_dict[hash_id]=modeldict # db only updated if attribute added
                    except:
                        self.logger.exception(f'error for hash_id:{hash_id},modeldict:{modeldict}!')
                        assert False, 'halt'
                    
                else:
                    assert False,'expecting cross_validate dict '

            rdb_dict.commit()
        return
            
            