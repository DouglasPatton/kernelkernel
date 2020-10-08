from mylogger import myLogger
from  sk_tool import SKToolInitializer

class FitRunner(myLogger):
    def __init__(self,rundict):
        myLogger.__init__(self,name='FitRunner.log')
        self.logger.info('starting FitRunner logger')
        self.rundict=rundict
    def passQ(self,saveq)
        self.saveq=saveq
    def run(self,):
        
        
        data,hash_id_model_dict=self.build_from_rundict(self.rundict)
        hash_id_list=list(hash_id_model_dict.keys())
                    shuffle(hash_id_list)# so all nodes aren't working on the same algorithm at startup
                    for hash_id in hash_id_list:
                        model_dict=hash_id_model_dict[hash_id]
                        try:
                            success=0
                            model_dict['model']=model_dict['model'].run(data)
                            success=1
                        except:
                            self.logger.exception('error for model_dict:{model_dict}')
                        savedict={hash_id:model_dict}
                        qtry=0
                        while success:
                            self.logger.debug(f'adding savedict to saveq')
                            try:
                                qtry+=1
                                self.saveq.put(savedict)
                                self.logger.debug(f'savedict sucesfully added to saveq')
                                break
                            except:
                                if not self.saveq.full() and qtry>3:
                                    self.logger.exception('error adding to saveq')
                                else:
                                    sleep(1)
                                    
                                    
                                    
    def build_from_rundict(self,rundict):
        data_gen=rundict['data_gen'] #how to generate the data
        data=dataGenerator(data_gen)
        model_gen_dict=rundict['model_gen_dict'] # {hash_id:data_gen...}
        hash_id_model_dict={}
        for hash_id,model_gen in model_gen_dict.items():
            model_dict={'model':SKToolInitializer(model_gen),'data_gen':data_gen,'model_gen':model_gen}
            hash_id_model_dict[hash_id]=model_dict# hashid based on model_gen and data_gen
        return data,hash_id_model_dict