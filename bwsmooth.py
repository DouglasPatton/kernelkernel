import numpy as np
import logging

class BWsmooth:
    def __init__(self,savedir=None,myname=None):
        if savedir==None:
            savedir=os.getcwd()
        self.savedir=savedir
        
        logging.basicConfig(level=logging.INFO)
        logdir=os.path.join(self.savedir,'log')
        if not os.path.exists(logdir): os.mkdir(logdir)
        handlername=f'BWsmooth.log'
        handler=logging.FileHandler(os.path.join(logdir,handlername))
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(handler)

        self.name=myname
        self.cores=int(psutil.cpu_count(logical=False)-1)
      


    