import logging
import numpy as np

'''from bokeh.io import show, output_notebook,curdoc,save, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Range1d, BBoxTileSource
from bokeh.layouts import row
output_notebook()
#output_file("map1.html",'bokeh graphs')'''
'''
import matplotlib.pyplot as plt
#%matplotlib inline
#the above line must be included and commented out
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

class DataPlotter:
    def __init__(self):
        self.logger=logging.getLogger(__name__)
        self.logger.debug('DataView object started')
        plt.rc_context({
            'figure.autolayout': True,
            'axes.edgecolor':'k', 
            'xtick.color':'k', 'ytick.color':'k', 
            'figure.facecolor':'grey'})
        self.colors=list('grkycbmk')
        self.hatches=['-', '+', 'x', '\\', '*', 'o', 'O', '.', '/']
        self.linestyles=[
            (0, (1, 10)),(0, (1, 1)),(0, (1, 1)),
            (0, (5, 10)),(0, (5, 5)),(0, (5, 1)),
            (0, (3, 10, 1, 10)),(0, (3, 5, 1, 5)),
            (0, (3, 1, 1, 1)),(0, (3, 5, 1, 5, 1, 5)),
            (0, (3, 10, 1, 10, 1, 10)),(0, (3, 1, 1, 1, 1, 1))]
        #self.fig=plt.figure(figsize=[14,80])
        
    def makePlotWithCI(self,x,y,std_err,ax,plottitle='',color='b',hatch='x',ls='-',lower=None,upper=None,ci_off=False):
        if type(color) is int:
            color=cm.Set1(color)
        if type(hatch) is int:
            hatch=self.hatches[hatch]
        if type(ls) is int:
            ls=self.linestyles[ls]
        k=len(x)
        self.logger.info(f'plotting type(x):{type(x)},x:{x}')
        self.logger.info(f'plotting type(y):{type(y)},x:{y}')
        self.logger.info(f'ls:{ls},plottitle:{plottitle},color:{color}')
        ax.plot(x,y,label=plottitle,color=color,alpha=0.7,linewidth=0.5,)#,ls,label=plottitle,color=color)
        if not ci_off:
            if not std_err is None:
                lower,upper=zip(*[(y[i]-std_err[i]*1.96,y[i]+std_err[i]*1.96) for i in range(k)])
            else: assert not (lower is None or upper is None), f'expected None but lower:{lower},upper:{upper}'
            ax.fill_between(x,lower,upper,alpha=0.01,color=color,hatch=5*hatch,)#label=f'95% CI for {plottitle}')
        #ax.plot(x,[0]*len(x),':',label='_0',color='k',alpha=0.5)
        #[ax.axvline(x=x[i],ls=':',alpha=0.3,label='_band',color='k') for i in range(k)]
        
        
          
    def my2dbarplot(self,x,y,xlabel='x',ylabel='y',title=None,fig=None,subplot_idx=(1,1,1)):
        if not fig: fig=plt.figure(dpi=self.dpi,figsize=[14,10])
        ax=fig.add_subplot(*subplot_idx)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        width=0.8*(x.max()-x.min())/x.size
        if title:
            ax.set_title(title)
        ax.bar(x,y,color='darkslategrey', alpha=0.8,width=width)
        ax.grid(True)
        '''self.ax.xaxis.label.set_color('blue')
        self.ax.yaxis.label.set_color('blue')                                  
        self.ax.spines['top'].set_color('blue') '''
        return fig
    
    def my2dscatter(self,x,x_name,y,y_name,ax=None,log_scale=False):
        if ax is None:
            plotter=plt
            plt_kwargs={'axes':ax}
        else:
            plotter=ax
            plt_kwargs={}
        plotter.scatter(x,y)
        plt.xlabel(x_name,**plt_kwargs)
        plt.ylabel(y_name,**plt_kwargs)
        if log_scale:
            plt.xscale('log',**plt_kwargs)
            plt.yscale('log',**plt_kwargs)
        if  ax is None:
            plotter.show()
            plotter.savefig(f'{x_name}_by_{y_name}_scatter.png')
    
    def my2dHist(self,data,name,ax=None,log_bins=False,bin_count=50):
        if type(data) is dict:
            data_=data[name]
        else:
            data_=data
        if ax is None:
            plotter=plt
            plt_kwargs={'axes':ax}
        else:
            plotter=ax
            plt_kwargs={}
        if log_bins:
            min_val=min(data_)
            max_val=max(data_)
            hist_kwargs={
                'bins':np.logspace(np.log10(min_val),np.log10(max_val),bin_count)}
        else:
            hist_kwargs={'bins':bin_count}
        n,bins,patches=plotter.hist(data_,density=False,**hist_kwargs)
        plt.xlabel(name,**plt_kwargs)
        plt.ylabel('count',**plt_kwargs)
        #plt.title(f'Histogram of {name}',**plt_kwargs)
        #plotter.text(60, .025, r'$\mu=100,\ \sigma=15$')
        #plotter.xlim(40, 160)
        #plotter.ylim(0, 0.03)
        plt.grid(True,**plt_kwargs)
        if log_bins:
            plt.xscale('log',**plt_kwargs)
        if  ax is None:
            plotter.show()
            plotter.savefig(f'{name}_dhist.png')
            
            
            
    def my3dHistogram(self,arraylist,varname,dim3list,subplot_idx=[1,1,1],fig=None,norm_hist=1,ylabel='time'): 
        if fig is None:
            fig=plt.figure(dpi=self.dpi,figsize=[self.figwidth,self.figheight])
            
        uniquecount=max([np.unique(nparray).size for nparray in arraylist])
        if uniquecount<4:nbins=8
        elif uniquecount<10:nbins=30
        else:nbins=min([50,uniquecount])
        maxrange=max([nparray.max() for nparray in arraylist])-min([nparray.min() for nparray in arraylist])
        #width=0.7*(np.log10(nbins**1.5)-1)*(maxrange)/nbins    
        width=.8*maxrange/nbins
        yarray=np.array(dim3list,dtype=int)
        left=min([nparray.min() for nparray in arraylist])
        right=max([nparray.max() for nparray in arraylist])
        #right=max([np.percentile(nparray,99) for nparray in arraylist])
        ax=fig.add_subplot(*subplot_idx,projection='3d')
        ax.set_xlabel(varname)
        ax.set_ylabel(ylabel)
        if norm_hist:
            ax.set_zlabel(f'relative frequency by {ylabel}')
            ax.set_zticklabels([])
        else: ax.set_zlabel(f'count per bin of {nbins} bins')
        ax.set_xlim(left,right)
        histtuplist=[]
        for i,nparray in enumerate(arraylist):
            y=yarray[i]
            freq,bins=np.histogram(nparray,bins=nbins,density=norm_hist)#dim0 is time
            x=0.5*(bins[1:]+bins[:-1])
            z=freq
            ax.view_init(elev=40,azim=-35)
            ax.bar(x,z,zs=y,zdir='y',alpha=0.95,width=width)
            #ax.bar(y,z,zs=x,zdir='x',alpha=0.95,width=width)
            histtuplist.append((freq,bins))

        return fig,{'varname':varname,'hist_tup_list':histtuplist,'norm_hist':norm_hist}
        
        

        
            
    def plotGeog(self,arraylist,yearlist=None):
        lat_idx=self.varlist.index('latitude')
        lon_idx=self.varlist.index('longitude')
        latlist=[nparray[:,lat_idx] for nparray in arraylist]
        longlist=[nparray[:,lon_idx] for nparray in arraylist]
        pass
        
