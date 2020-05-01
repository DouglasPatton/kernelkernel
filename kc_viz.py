from kernelcompare import KernelCompare
import os
import logging


class KCViz(KernelCompare,data_viz):
    def __init__(self,directory=None):
        try:
            self.logger=logging.getLogger(__name__)
            self.logger.info('starting new kc_viz object')
        except:
            logdir=os.path.join(directory,'log')
            if not os.path.exists(logdir): os.mkdir(logdir)
            handlername=os.path.join(logdir,__name__)
            logging.basicConfig(
                handlers=[logging.handlers.RotatingFileHandler(handlername, maxBytes=10**7, backupCount=100)],
                level=logging.WARNING,
                format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                datefmt='%Y-%m-%dT%H:%M:%S')
            self.logger = logging.getLogger(handlername)
            self.logger.info(f'starting new {handlername} log')
        if directory is None:
            directory=os.path.join(os.getcwd(),'results')
        self.directory=directory
        KernelCompare.__init__(self,directory=self.directory,source='pisces')
        data_viz.__init__(self)