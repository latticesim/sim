"""
Simulation of clonal evolution on a hexagonal lattice
Interfaces with sim.cpp via ctypes 

Simulation of neutral competition: model_neutral()
Simulation of survival advantage: model_die()
Rendering of video: movie_die()
Real time rendering of model evolution: RealTime class
Migration with neutral drift: model_neutral_migration()
Heterogeneous stem cell compartment: model_stem_ta()
Multiple cells at each lattice point: model_3d()

M Lynch 2017
"""

import ctypes
from numpy.ctypeslib import ndpointer
import numpy as np
import cPickle as pickle
import sys, socket
from copy import deepcopy

HOST = socket.gethostname() 
if HOST!='vmgamma': #local
    PATH =  '/programs/'
    RESULTS = '/vol/results/clones/'
    FIGS = '/capseq/analysed/figs/'
else: #server
    PATH = '/media/sf_Programs/'
    RESULTS = '/media/sf_magnuslynch/Files/Valuable/Research/clones/' 
    FIGS = '/media/sf_magnuslynch/Dropbox/Current Work/Research/Watt Lab/Figures/Epidermis Capseq/' 

sys.path.append(PATH+'lab/python/')
lib = ctypes.CDLL(PATH+'lab/cpp/sim/sim.so')


CALLBACK = ctypes.CFUNCTYPE(None,ctypes.c_int)
PALETTE = '#538ecb,#212121,#e48782,#c0c0c0,#7fba7e,#f9d27f'.split(',')


def set_default(thedict,key,val):
    if not thedict.has_key(key): thedict[key] = val 

import clones_plot


class Model(ctypes.Structure):
    """
    ctypes interface to the cpp Model class
    """
    _fields_ = [
        ("p_mut", ctypes.c_float),
        ("p_diff", ctypes.c_float),
        ("p_diff_ta", ctypes.c_float),
        ("rep_hi", ctypes.c_float),
        ("rep_lo", ctypes.c_float),
        ("rep_mut", ctypes.c_float),
        ("die_hi", ctypes.c_float),
        ("die_lo", ctypes.c_float),
        ("die_mut", ctypes.c_float),
        ("stem_mut_rate", ctypes.c_float),
        ("multi_mut", ctypes.c_float),
        ("multi_mut_mean", ctypes.c_float),
        ("multi_mut_sd", ctypes.c_float),
        ("migration_stem", ctypes.c_float),
        ("migration_ta", ctypes.c_float),
        ("niche_radius", ctypes.c_float),
        ("niche_sep", ctypes.c_float),
        ("ta_max_reps", ctypes.c_int),
        ("callback_freq", ctypes.c_int)]

model = Model.in_dll(lib,"py_model")   


def hex_grid(v,mode='mut',mut=None,ax=None):
    """
    Render cells on the hexagonal lattice
    """
    #print 'Plotting...'
    nx = v.shape[0] #100 
    ny = v.shape[1] #50
    radius = 10 

    import pylab as pl
    #pl.clf()
    if ax is None: 
        fig = pl.figure(figsize=(8,7))
        ax = pl.gca()
    else:
        ax.set_xlim((0,8))
        ax.set_ylim((8,7))

    #cmap = pl.cm.jet
    cmap = pl.cm.gist_rainbow
    cmap = [cmap(i) for i in range(cmap.N)]
    
    gapx = radius*2 
    gapy = (gapx/2.0) / np.tan(np.pi/6.0)
    for ix in xrange(0,nx):
        for iy in xrange(0,ny):
            x = (float(ix)+0.5)*gapx
            y = (float(iy)+0.5)*gapy
            if iy % 2 == 0: x += 0.5 * gapx
            fill=False
            if v[ix][iy] > 0: fill = True 
            color='b'
            fill=True
            if mode=='multiple': 
                color=(255,255,255)
                if v[ix][iy]>=0: 
                    index = int(v[ix][iy]) % len(cmap)
                    color=cmap[index]
            elif mode=='single':
                if v[ix][iy] <= 0: fill = False 
            elif mode=='mut':
                if mut[ix][iy] > 0: 
                    index = int(v[ix][iy]) % len(cmap)
                    color=cmap[index]
                else:
                    fill = False 
            else: assert False
            
            if mode != 'mut' or fill != False:
                circ = pl.Circle((x,y), radius=radius, fill=fill, color=color)
                ax.add_patch(circ)
        
    ax.set_xlim(0,float(nx+0.5)*gapx)
    ax.set_ylim(0,float(ny)*gapy)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def text_grid(v):
    """
    Show text mode representation of cells on hexagonal lattice
    """
    v = np.rot90(v)
    maxstring = len(str(np.max(v)))
    for iy in xrange(v.shape[1]):
        w = v[iy]
        w[w<0] = -1
        s = [str(int(x)) for x in w]
        s = [x.ljust(maxstring) for x in s]
        s = '|'.join(s)
        s=s.replace('-1','.'.ljust(maxstring))
        print s 
    print '=================='

def iter_grid(v):
    for ix in xrange(v.shape[0]):
        for iy in xrange(v.shape[1]):
            yield ix,iy


class RealTime:    
    """
    Render evolution of model in real time using pygame
    """
    def __init__(self,running=False,width=600):
        import pygame as pg
        pg.init()
        #width=600
        height=int(self.height_ratio(width))
        self.window = pg.display.set_mode((width,height))
        pg.display.set_caption('Sim')
        surface = pg.Surface((width,height))
        
        #Start the event loop
        while running: self.handle_events()
   
    def handle_events(self):
        import pygame as pg
        events = pg.event.get()
        if len(events)>0: 
            for event in events:
                if event.type==pg.QUIT: 
                    #running=False
                    self.quit()
    
    def quit(self):
        import pygame as pg
        print 'Quitting...1'
        pg.display.quit()
        print 'Quitting...2'
        import os; os._exit(0) 

    def height_ratio(self,x):
        """
        Return the height of a hexagonal lattice given the width
        """
        return (x/2.0) / np.tan(np.pi/6.0)

    def draw_cells(self,cells,mode='single',mut=None,cell_type=None,reps=None):
        """
        Draw the hexagonal grid of cells on the pygame window
        """
        import pygame as pg
        from pygame import gfxdraw
        import pylab as pl

        def cmap_to_pygame(cmap):
            return np.array([cmap(i)[:3] for i in range(cmap.N)])*255.0
        #cmap = pl.cm.gist_rainbow
        #cmap = np.array([cmap(i)[:3] for i in range(cmap.N)])*255.0
        cmap = cmap_to_pygame(pl.cm.gist_rainbow)
        reds = cmap_to_pygame(pl.cm.autumn)
        blues = cmap_to_pygame(pl.cm.winter)
        greens = cmap_to_pygame(pl.cm.Greens)
        jet = cmap_to_pygame(pl.cm.jet)
        width,height = self.window.get_size()
        surface = pg.Surface((width,height))
        nx = cells.shape[0] 
        ny = cells.shape[1]
        assert nx==ny
        radius = float(width) / float(nx*2) 
        gapx = float(radius*2)
        gapy = self.height_ratio(gapx)
        
        for ix in xrange(0,nx):
            for iy in xrange(0,ny):
                x = (float(ix)+0.5)*gapx
                y = (float(iy)+0.5)*gapy
                if iy % 2 == 0: x += 0.5 * gapx
                if mode=='single':
                    color=(0,0,0)
                    if cells[ix][iy] > 0: color=(255,255,255) 
                elif mode=='multiple':
                    color=(255,255,255)
                    if cells[ix][iy]>0: 
                        index = int(cells[ix][iy]) % len(cmap)
                        color=cmap[index]
                elif mode=='mut':
                    if mut[ix][iy] > 0:
                        index = int(cells[ix][iy]) % len(cmap)
                        color=cmap[index]
                    else: 
                        color=(255,255,255)
                elif mode=='cell_type_mut':
                    if mut[ix,iy]>0: 
                        color = reds[int(cells[ix][iy]) % len(greens)]
                    elif cell_type[ix,iy]==0: 
                        #color = greens[int(cells[ix][iy]) % len(reds)]
                        color = (0,255,0)
                    elif cell_type[ix,iy]>0: 
                        #color = blues[int(cells[ix][iy]) % len(blues)]
                        color = (0,0,255)
                    else: 
                        color=(255,255,255)
                elif mode=='cell_type':
                    #stem red, ta blue
                    if cell_type[ix][iy] < 0: color = (255,255,255);
                    elif cell_type[ix][iy] > 0: color = blues[int(cells[ix][iy]) % len(blues)]
                    else: color = greens[int(cells[ix][iy]) % len(reds)]
                elif mode=='cell_type_simple':
                    #stem red, ta blue
                    if mut is not None and mut[ix,iy]>0: color = (255,0,0)
                    elif cell_type[ix][iy] < 0: color = (255,255,255)
                    elif cell_type[ix][iy] > 0: color=(0,0,255) 
                    else: color=(0,255,0)
                elif mode=='reps':
                    r = reps[ix][iy]
                    if cell_type[ix][iy] < 0: color = (255,255,255)
                    else: 
                        i = float(reps[ix][iy])/float(np.max(reps)+1)
                        color = jet[int(i*len(jet))]
                    #elif r==1: color = (255,0,0) 
                    #elif r==2: color = (0,255,0) 
                    #elif r==3: color = (0,0,255) 
                    #else: color = (0,0,0)

                else: assert False
                pg.draw.circle(surface,color,(int(x),int(y)),int(radius))
        
        #Copy to display 
        surface = pg.transform.flip(surface,False,True) #flip to match hexgrid
        self.window.blit(surface, (0,0))
        pg.display.flip()


    def save(self,filename):
        """
        Save an image of the current window
        """
        import pygame as pg
        pg.image.save(self.window,filename)


def realtime_test():
    cells = pickle.load(open(RESULTS+'realtime_plot.pkl','rb'))
    #import pylab as pl
    #hex_grid(cells,mode='single')
    #pl.savefig(FIGS+'realtime_test.jpg')
    
    realtime = RealTime() 
    realtime.draw_cells(cells)
    while True: realtime.handle_events()

    
def array_from_cpp(v):
    n = int(np.sqrt(v.shape[0]))
    v = np.reshape(v,(n,n))
    v=v.transpose() 
    return v



def make_grid_fig():
    """
    Make a figure of a 20x20 hexagonal lattice for use as a figure
    """
    v = np.zeros((10,10))
    v[5,5] = 1
    import pylab as pl
    hex_grid(v,mode='single')
    pl.savefig('/capseq/analysed/figs/grid_fig.eps')


def get_int_buffer(nbuffer=0):
    """
    Get the contents of the int buffer from cpp module
    then clear the buffer
    """
    n = lib.get_int_buffers_size(nbuffer)
    if n==0: return np.array([]) 
    lib.get_int_buffers.restype = ndpointer(dtype=ctypes.c_int,shape=(n,))
    data = lib.get_int_buffers(nbuffer)
    lib.free_int_buffers(nbuffer)
    return data

def get_float_buffer(nbuffer=0):
    """
    Get the contents of the float buffer from cpp module
    then clear the buffer
    """
    n = lib.get_float_buffers_size(nbuffer)
    if n==0: return np.array([]) 
    lib.get_float_buffers.restype = ndpointer(dtype=ctypes.c_float,shape=(n,))
    data = lib.get_float_buffers(nbuffer)
    lib.free_float_buffers(nbuffer)
    return data

def get_array():
    #Get array of cells from cpp module
    #then clear the buffer
    data = get_int_buffer()
    return array_from_cpp(data) 


def model_neutral(modes=['calc','plot'],n=3,size=200,generations=1000,vals=[0.01]):
    """
    Simulation of neutral competition
    """

    def callback(generation):
        clones = get_float_buffer()
        cells = get_array()
        print 'called',generation,len(clones)

  
    reps = {} #for each val run n times to get error bars

    if 'calc' in modes:
        for p_mut in vals:
            print p_mut,'-'*60
            reps[p_mut] = []
            for i in xrange(n): 
                print 'rep',i
                sys.stdout.flush()
                lib.py_create_model(size)
                model.p_mut = p_mut; 
                model.callback_freq = 10;
                lib.py_model_neutral(generations,CALLBACK(callback))
                clones = get_float_buffer(); 
                clones = sorted(clones,reverse=True)
                reps[p_mut].append(clones)
                print len(clones) 
        
        pickle.dump(reps,open(RESULTS+'model_neutral.pkl','wb'))
    
    else: reps = pickle.load(open(RESULTS+'model_neutral.pkl','rb'))
   
    if 'plot' in modes:
        import pylab as pl
        for val,r in reps.iteritems():
            r.sort(reverse=True)
            print r[0][:10]
            pl.clf()
            clones_plot.plot_multi(r,log=True)
            filename = FIGS+'model_neutral_log'+str(val)+'.eps'
            pl.savefig(filename)
            #pl.show()
            pl.clf()
            clones_plot.plot_multi(r,log=False)
            filename = FIGS+'model_neutral'+str(val)+'.eps'
            pl.savefig(filename)
            #pl.show()





def model_die(modes=['calc','plot'],generations=200,cutoff=0.0,log=False,n=10,size=200,**kwargs):
    """
    Simulation of survival advantage 
    Vary the selective advantage of mutant clone
    """
    set_default(kwargs,'p_mut',0.001)
    set_default(kwargs,'die_hi',1.0)
    set_default(kwargs,'die_lo',0.5)
    set_default(kwargs,'migration',0.0)
    set_default(kwargs,'filename',RESULTS+'model_die_'+str(generations)+'_'+str(size))
    
    if 'realtime' in modes: realtime = RealTime(width=900) 
    
    reps={}
    datafile = RESULTS + kwargs['filename'] + '.pkl'
    print datafile
    
    def callback(generation):
        if generation==generations: return
        
        print generation,
        sys.stdout.flush()
        cells = get_array()
        clones_hi = get_float_buffer(0)
        clones_lo = get_float_buffer(1)
        clones_all = get_float_buffer(2)
        mut = array_from_cpp(get_float_buffer(3))
        mut[mut==model.die_hi] = 0
        mut[mut==model.die_lo] = 1
        largest = sorted(clones_all,reverse=True)[:5]
        largest = [str(x) for x in largest]

        if 'realtime' in modes:
            realtime.draw_cells(cells,mode='mut',mut=mut)
            realtime.handle_events()
            if generation % 100 ==0: print '\nLargest Clones:'+ ','.join(largest)

    if 'calc' in modes:
        for die_mut in kwargs['die_mut']:
            print die_mut,'-'*60
            reps[die_mut] = []
            for i in xrange(n): 
                print 'rep',i
                lib.py_create_model(size)
                model.p_mut = kwargs['p_mut']
                model.die_mut = die_mut 
                model.die_hi = kwargs['die_hi'] #1.0
                model.die_lo = kwargs['die_lo'] #0.5
                model.migration_stem = kwargs['migration']
                if 'realtime' in modes:
                    model.callback_freq = 10
                else:
                    model.callback_freq = 100
                if 'migration' in modes: model.migration_stem = 1.0
                lib.py_model_die(generations,CALLBACK(callback))
                clones_hi = get_float_buffer(0); 
                clones_lo = get_float_buffer(1); 
                clones_all = get_float_buffer(2); 
                reps[die_mut].append((clones_hi,clones_lo,clones_all))
                print 'done'
                pickle.dump(reps,open(datafile,'wb'))
    
    else: reps = pickle.load(open(datafile,'rb'))

    if 'plot' in modes:
        #log=False
        for val,r in reps.iteritems():
            #if not val in vals: continue
            a=[]; b=[]; c=[]
            for x in r: 
                a.append(x[0])
                b.append(x[1])
                c.append(x[2])
            import pylab as pl
            #cutoff = 0.007
            #cutoff = 0.0
            pl.clf()
            clones_plot.plot_multi(a,cutoff=cutoff,col1=PALETTE[2],col2=PALETTE[2],log=log)
            clones_plot.plot_multi(b,cutoff=cutoff,col1=PALETTE[3],col2=PALETTE[3],log=log)
            clones_plot.plot_multi(c,cutoff=cutoff,col1=PALETTE[0],col2=PALETTE[0],log=log)
            xmin,xmax = pl.xlim()
            pl.xlim(cutoff,xmax)
            if not log:
                ymin,ymax = pl.ylim(); 
                pl.ylim(0,ymax)
            #filename = FIGS+'model_die'+str(generations)+'_'+str(val)+'_'+str(log)+'_'+str(cutoff)+'.eps'
            filename = FIGS+kwargs['filename']+str(val)+str(log)+str(cutoff)+'.eps'
            pl.savefig(filename)
            #pl.show()




def model_movie_die(modes=['calc','plot']):
    """
    Generate frames to be combined into a movie showing evolution of clones over time
    X-server fails randomly when plotting so save all stages first 
    """
    import os,sys
    size = 200
    p_mut = 0.01
    die_mut = 0.001
    generations = 3650 
    nframes = 1000 
    cutoff = 0.007
    """ 
    size = 200
    p_mut = 0.01
    die_mut = 1.0 
    generations = 100 
    nframes = min(3,generations)
    cutoff = 0.0 #smallest clone that can be detected by sequencing
    """ 
    data = []
    datafile = RESULTS+'movie_die.pkl'

    def movie_die_callback(generation):
        if nframes==1 and generation==0: return 
        clones_hi = get_float_buffer(0)
        clones_lo = get_float_buffer(1)
        clones_all = get_float_buffer(2)
        mut = array_from_cpp(get_float_buffer(3))
        mut[mut==model.die_hi] = 0
        mut[mut==model.die_lo] = 1
        cells = get_array() 
        detectable = np.extract(clones_all>cutoff,clones_all)
        X,Y = ca.counts_hist(detectable)
        data.append((generation,cells,mut,X,Y))
        print 'called',generation
        sys.stdout.flush()

    callback = CALLBACK(movie_die_callback)
 
    if 'calc' in modes: 
        lib.py_create_model(size)
        model.p_mut = p_mut 
        model.die_mut = die_mut 
        model.callback_freq = generations / nframes 
        lib.py_model_die(generations+1,callback)
        pickle.dump(data,open(datafile,'wb'))

    if 'plot' in modes:
        import matplotlib as mpl
        #Need to use 'agg' otherwise fails when x server logs off
        mpl.use('Agg')
        import matplotlib.pyplot as pl 
        #import pylab as pl
        
        print 'loading...'
        data = pickle.load(open(datafile,'rb'))
        frame = 0  

        for frame, v in enumerate(data):
            generation,cells,mut,X,Y = v
            filename = ca.FIGS+'vid/movie%05d.jpg'%frame
            if os.path.exists(filename): 
                print 'skipping:',filename
                sys.stdout.flush()
                continue

            pl.clf()
            fig,axes = pl.subplots(nrows=1,ncols=2,figsize=(16,7))
            hex_grid(cells,mode='mut',mut=mut,ax=axes[0])
            if len(Y)>2:
                X,Y = ca.simons_transform(X,Y)
                fit = ca.simons_fit(X,Y)
                ca.simons_plot(X,Y,fit,log=False,col1='k',col2='k',ax=axes[1]);
            axes[1].text(0.5, 0.9,str(generation)+' days',horizontalalignment='center',verticalalignment='center',transform=axes[1].transAxes)
            pl.tight_layout()
            pl.savefig(filename)
            print filename
            sys.stdout.flush()
            pl.close(fig)



def movie_frames_plot(modes=['extract','plot']): 
    """
    Plot selected frames from a movie in eps for inclusion in a figure
    """
    path = '/vol/results/clones/'
    data = path+'run3/movie_die.pkl'
    frames = [483,905] 
    temp_file = path+'movie_frames_plot.pkl'

    if 'extract' in modes:
        data = pickle.load(open(data,'rb'))
        temp = []
        for frame in frames: temp.append(data[frame])
        pickle.dump(temp,open(temp_file,'wb'))

    if 'plot' in modes:
        temp = pickle.load(open(temp_file,'rb'))
        for generation,cells,mut,X,Y in temp:
            print generation
            import pylab as pl
            pl.clf()
            hex_grid(cells,mode='mut',mut=mut,ax=None)
            pl.savefig(FIGS+'movie_frames_hex_'+str(generation)+'.jpg')
            pl.clf()
            if len(Y)>2:
                X,Y = clones_plot.transform(X,Y)
                fit = clones_plot.fit(X,Y)
                clones_plot.plot(X,Y,fit,log=False,col1='k',col2='k');
                pl.savefig(FIGS+'movie_frames_simons_'+str(generation)+'.eps')





 
def model_neutral_realtime():
    """
    Visualize the evolution of cells in neutral drift in real time
    """
    import pylab as pl
    realtime = RealTime() 
    
    def realtime_plot_callback(generation):
        clones = get_float_buffer()
        cells = get_array()
        realtime.draw_cells(cells)
        realtime.handle_events()

    callback = CALLBACK(realtime_plot_callback)
    

    lib.py_create_model(100)
    model.callback_freq = 1 
    model.p_mut = 0.00005
    lib.py_model_neutral(500,callback)
    
    while True: realtime.handle_events()

    
def model_neutral_migration(modes=['calc','save']):
    """
    What are the consequences of cellular migration for clustering and 
    clone size distributions arising from neutral drift 
    """
    if 'realtime' in modes:
        realtime = RealTime() 
    
    def callback(generation):
        global clones
        clones = get_float_buffer()
        cells = get_array()
        if 'realtime' in modes:
            realtime.draw_cells(cells,mode='multiple')
            realtime.handle_events()
        print generation,
        sys.stdout.flush()
       
    vals = [1.0,0.1,0.01,0.001,0.0]
    reps = {} 
        
    if 'calc' in modes:
        for val in vals:
            for rep in xrange(3): 
                print '\nval',val,
                #lib.py_create_model(200)
                lib.py_create_model(100)
                model.p_mut = 0.01
                model.callback_freq = 1
                model.migration_stem = val 
                lib.py_model_neutral(1000,CALLBACK(callback))
                cells = get_array() 
                clones = get_float_buffer()
                if not reps.has_key(val): reps[val] = []
                reps[val].append((clones,cells))
        
        if 'save' in modes: pickle.dump(reps,open(RESULTS+'neutral_migration.pkl','wb'))
        if 'realtime' in modes: realtime.quit() 
   
    if 'plot' in modes:
        import pylab as pl
        print 'plotting'
        reps = pickle.load(open(RESULTS+'neutral_migration.pkl','rb'))
        print reps.keys()
        for val in reps.keys():
            print val
            v = [x[0] for x in reps[val]]
            #Plot clone size distributions
            pl.clf()
            clones_plot.plot_multi(v,log=False)
            pl.savefig(FIGS+'neutral_migration_'+str(val)+'.eps')
        
            #Plot example of lattice 
            pl.clf()
            hex_grid(reps[val][0][1],mode='multiple')
            pl.savefig(FIGS+'neutral_migration_lattice'+str(val)+'.jpg')

            
def model_stem_ta(modes,**kwargs):
    """
    Simulation of heterogeneous stem cell compartment
    """
    set_default(kwargs,'p_mut',0.001)
    set_default(kwargs,'ta_max_reps',14)
    set_default(kwargs,'stem_mut',0.0)
    set_default(kwargs,'migration_ta',0.2)
    set_default(kwargs,'radius',5.0)
    set_default(kwargs,'separation',30.0)
    set_default(kwargs,'draw_mode','cell_type_simple')
    set_default(kwargs,'callback_freq',20)

    if 'realtime' in modes or 'draw' in modes: realtime = RealTime(width=900)

    if 'record' in modes and 'calc' in modes: 
        record = []
        assert kwargs['reps']==1

    def from_model():
        results = {}
        results['cells'] = get_array() 
        results['cell_type'] = array_from_cpp(get_int_buffer(nbuffer=1))
        results['reps'] = array_from_cpp(get_int_buffer(2))
        results['mut'] = array_from_cpp(get_int_buffer(3))
        results['clones_hi'] = get_float_buffer(0)
        results['clones_lo'] = get_float_buffer(1)
        results['clones_all'] = get_float_buffer(2)
        return results 


    def callback(generation):
        print generation,
        sys.stdout.flush()
        results = from_model() 

        if 'realtime' in modes:
            realtime.draw_cells(results['cells'],mode=kwargs['draw_mode'],cell_type=results['cell_type'],reps=results['reps'],mut=results['mut'])
            realtime.handle_events()
        if 'realtime-draw' in modes:
            realtime.save(RESULTS+'video/'+kwargs['filename']+str(generation)+'.png')



        if 'record' in modes: record.append((generation,results))

    if 'calc' in modes:
        results = [] 
        
        for i in xrange(kwargs['reps']):
            print '\nRep',i,'='*80

            #sep = 30.0
            #rad = 5.0 
            size = (int)(kwargs['separation'] * kwargs['clusters'])
            lib.py_create_model(size)
            model.p_mut = kwargs['p_mut'] 
            model.p_diff = 0.5/7.0 #rate of stem cell loss 
            #model.p_diff_ta = model.p_diff*2.0 #rate of differentiated cell loss
            model.p_diff_ta = model.p_diff #rate of differentiated cell loss
            model.migration_stem = 0.0
            model.migration_ta = kwargs['migration_ta'] 
            model.die_hi = 1.0
            model.die_lo = 0.5
            model.stem_mut_rate = kwargs['stem_mut'] 
            model.callback_freq = kwargs['callback_freq'] 
            model.niche_radius = kwargs['radius'] 
            model.niche_sep = kwargs['separation'] 
            model.ta_max_reps = kwargs['ta_max_reps'] 
           
            lib.py_model_stem_ta(kwargs['gens'],CALLBACK(callback))

            v = deepcopy(kwargs)
            v['rep'] = i
            results.append((v,from_model()))
            if 'record' in modes: record.append((kwargs['gens'],results))
        
        print 'saving...'
        pickle.dump(results,open(RESULTS+kwargs['filename']+'.pkl','wb'))
        if 'record' in modes:
            pickle.dump(record,open(RESULTS+kwargs['filename']+'_record.pkl','wb'))


    if 'plot' in modes or 'draw' in modes:
        results = pickle.load(open(RESULTS+kwargs['filename']+'.pkl','rb'))
        print results
        if 'record' in modes: 
            record = pickle.load(open(RESULTS+kwargs['filename']+'_record.pkl','rb'))
        
    if 'plot' in modes:
        import pylab as pl
        pl.clf()
        clones = [x[1]['clones_all'] for x in results]
        clones_plot.plot_multi(clones,log=False,cutoff=kwargs['cutoff'])
        lims = pl.ylim()
        pl.ylim(0.0,lims[1])
        lims = pl.xlim()
        pl.xlim(0.0,lims[1])
        pl.savefig(FIGS+kwargs['filename']+'.eps')
        pl.show()

    if 'draw' in modes:
        if 'record' in modes:
            print len(record)
            for x in record:
                print x[0]
                realtime.draw_cells(x[1]['cells'],mode=kwargs['draw_mode'],cell_type=x[1]['cell_type_mut'],reps=x[1]['reps'],mut=x[1]['mut'])
                realtime.save(FIGS+kwargs['filename']+str(x[0])+'.png')

        else:
            realtime.draw_cells(results[0][1]['cells'],mode=kwargs['draw_mode'],cell_type=results[0][1]['cell_type'],reps=results[0][1]['reps'],mut=results[0][1]['mut'])
            realtime.save(FIGS+kwargs['filename']+'_'+kwargs['draw_mode']+'r.png')
            realtime.draw_cells(results[0][1]['cells'],mode='cell_type_simple',cell_type=results[0][1]['cell_type'],reps=results[0][1]['reps'],mut=results[0][1]['mut'])
            realtime.save(FIGS+kwargs['filename']+'_'+kwargs['draw_mode']+'s.png')
            #while True: realtime.handle_events()

    while 'realtime' in modes: realtime.handle_events()

   
   

def model_3d(modes=['calc','plot'],generations=200,n=10,size=200,height=1,**kwargs):
    """
    Simulation of survival advantage 
    Vary the selective advantage of mutant clone
    """
    set_default(kwargs,'p_mut',0.001)
    set_default(kwargs,'die_hi',1.0)
    set_default(kwargs,'die_lo',0.5)
    set_default(kwargs,'die_mut',[0.0])
    set_default(kwargs,'migration',0.0)
    set_default(kwargs,'cutoff',0.007)
    set_default(kwargs,'log',False)
    set_default(kwargs,'filename','model_3d_'+str(generations)+'_'+str(size))
    
    if 'realtime' in modes: realtime = RealTime(width=900) 
    
    reps={}
    level = 1 #which level in the 3D lattice to plot
    datafile = RESULTS + kwargs['filename'] + '.pkl'
    print datafile
    
    def callback(generation):
        if generation==generations: return
        
        print generation,
        sys.stdout.flush()
        #cells = get_array()
        cells = get_int_buffer(0)
        cells = np.reshape(cells,(height,size,size))
        clones_hi = get_float_buffer(0)
        clones_lo = get_float_buffer(1)
        clones_all = get_float_buffer(2)
        #mut = array_from_cpp(get_float_buffer(3))
        mut = get_float_buffer(3)
        mut = np.reshape(mut,(height,size,size))
        mut[mut==model.die_hi] = 0
        mut[mut==model.die_lo] = 1
        largest = sorted(clones_all,reverse=True)[:5]
        largest = [str(x) for x in largest]

        if 'realtime' in modes:
            realtime.draw_cells(cells[level],mode='mut',mut=mut[level])
            realtime.handle_events()
            if generation % 100 ==0: print '\nLargest Clones:'+ ','.join(largest)

    if 'calc' in modes:
        for die_mut in kwargs['die_mut']:
            print die_mut,'-'*60
            reps[die_mut] = []
            for i in xrange(n): 
                print 'rep',i
                lib.py_create_model_3d(size,height)
                model.p_mut = kwargs['p_mut']
                model.die_mut = die_mut 
                model.die_hi = kwargs['die_hi'] #1.0
                model.die_lo = kwargs['die_lo'] #0.5
                model.migration_stem = kwargs['migration']
                if 'realtime' in modes:
                    model.callback_freq = 10
                else:
                    model.callback_freq = 100
                if 'migration' in modes: model.migration_stem = 1.0
                lib.py_model_3d(generations,CALLBACK(callback))
                clones_hi = get_float_buffer(0); 
                clones_lo = get_float_buffer(1); 
                clones_all = get_float_buffer(2); 
                reps[die_mut].append((clones_hi,clones_lo,clones_all))
                print 'done'
                pickle.dump(reps,open(datafile,'wb'))
    
    else: reps = pickle.load(open(datafile,'rb'))

    if 'plot' in modes:
        #log=False
        for val,r in reps.iteritems():
            #if not val in vals: continue
            a=[]; b=[]; c=[]
            for x in r: 
                a.append(x[0])
                b.append(x[1])
                c.append(x[2])
            import pylab as pl
            #cutoff = 0.007
            #cutoff = 0.0
            pl.clf()
            clones_plot.plot_multi(a,cutoff=kwargs['cutoff'],col1=PALETTE[2],col2=PALETTE[2],log=kwargs['log'])
            clones_plot.plot_multi(b,cutoff=kwargs['cutoff'],col1=PALETTE[3],col2=PALETTE[3],log=kwargs['log'])
            clones_plot.plot_multi(c,cutoff=kwargs['cutoff'],col1=PALETTE[0],col2=PALETTE[0],log=kwargs['log'])
            xmin,xmax = pl.xlim()
            pl.xlim(kwargs['cutoff'],xmax)
            if not kwargs['log']:
                ymin,ymax = pl.ylim(); 
                pl.ylim(0,ymax)
            #filename = FIGS+'model_die'+str(generations)+'_'+str(val)+'_'+str(log)+'_'+str(cutoff)+'.eps'
            filename = FIGS+kwargs['filename']+str(val)+str(kwargs['log'])+str(kwargs['cutoff'])+'.eps'
            #pl.show()
            pl.savefig(filename)


   
   


def make_figs():
    """
    Make all figures for the paper
    """
    pass
    #==========================================================
    #Stem TA
    model_stem_ta(modes=['calc'],clusters=2,reps=2,gens=100,stem_mut=0.001,filename='stem_ta_adv') 
    model_stem_ta(modes=['plot'],filename='stem_ta_adv') 
   
    radius = 20; separation = 45; gens=365*10 
    #radius = 5; separation = 30 

    #Competitive advantage
    #clone size dist
    filename='stem_ta_adv_'+str(radius)+'_'
    model_stem_ta(modes=['calc','realtime'],clusters=7,reps=10,gens=gens,stem_mut=0.001,filename=filename,radius=radius,separation=separation)
    model_stem_ta(modes=['draw','realtime'],filename=filename,draw_mode='cell_type_mut') 
    model_stem_ta(modes=['plot'],filename=filename,cutoff=0.007) 
    model_stem_ta(modes=['calc','realtime','realtime-draw'],clusters=7,reps=1,gens=3650,stem_mut=0.001,radius=radius,separation=separation,callback_freq=1,draw_mode='cell_type_mut',filename=filename)
    model_stem_ta(modes=['draw','record'],filename=filename,draw_mode='cell_type_mut')
    
    #Neutral evolution
    filename='stem_ta_neut'+str(radius)
    model_stem_ta(modes=['calc'],clusters=7,reps=10,gens=gens,stem_mut=0.0,radius=radius,separation=separation,filename=filename)
    model_stem_ta(modes=['plot'],filename=filename,cutoff=0.0) 
    model_stem_ta(modes=['draw'],filename=filename,draw_mode='cell_type_simple')

    
    #Effects of TA max reps
    for x in [1,2,5,10,15,20]:
        model_stem_ta(modes=['calc','draw'],clusters=2,reps=1,gens=100,ta_max_reps=x,migration_ta=0.0,filename='stem_ta_max_reps'+str(x),draw_mode='reps') 
    
    #Varying migration of TA cells
    for x in [0.0,0.1,0.5,0.7,1.0]:
        model_stem_ta(modes=['calc','draw'],clusters=2,reps=1,gens=100,ta_max_reps=15,migration_ta=x,filename='stem_ta_migrat_ta'+str(x),draw_mode='reps') 

    #Varying migration of stem cells

    #Varying size of stem cell clusters

    #==========================================================
    #Neutral
    #clone size dist
    #image

    #==========================================================
    #Boundary expansion

    model_die(['calc'],generations=365*10,size=200,cutoff=0.007,log=False,vals=[0.001,0.0],n=10)
    model_die(['plot'],vals=[0.001,0.0],generations=365*10,size=200,cutoff=0.007,log=False)

    #Magnitude of competitive advantage
    vals = [0.9,0.7,0.5,0.3,0.1,0.01]
    for x in vals: 
        model_die(['calc'],generations=365*10,size=100,cutoff=0.007,log=False,n=5,die_mut=[0.001],die_lo=x,filename=RESULTS+'model_die_adv'+str(x)+'.pkl')
        model_die(['plot'],cutoff=0.007,log=False,n=5,filename='model_die_adv'+str(x))

    #Effects of cell migration
    for x in [1.0,0.1,0.01,0.001,0.0]: 
        model_die(['calc','realtime'],generations=365*10,size=100,n=5,die_mut=[0.001],migration=x,filename=RESULTS+'model_die_mig'+str(x)+'.pkl')
        model_die(['plot'],cutoff=0.007,log=False,n=5,filename='model_die_mig'+str(x))

    
    
    #==========================================================
    #Neutral migration
    model_neutral_migration(modes=['calc','save','plot'])
    model_neutral_migration(modes=['plot'])
    
    #==========================================================
    #Old Sup. Fig. 5
    model_neutral(modes=['calc'],generations=365*3,size=200,vals=[0.1,0.01,0.001,0.0001],n=10) 
    model_neutral(modes=['plot']) 

    #==========================================================
    #Old Sup. Fig. 7
    model_die(['calc'],generations=365*10,size=200,die_mut=[1.0,0.1,0.01,0.001],n=10,filename='sup_fig7')
    model_die(['plot'],filename='sup_fig7',log=False)
    model_die(['plot'],filename='sup_fig7',cutoff=0.007,log=False)
    model_die(['plot'],filename='sup_fig7',cutoff=0.007,log=True)
    
    
    #==========================================================
    #Model 3D
    for n in xrange(1,7):
        model_3d(['plot'],n=5,size=200,height=n,generations=1000,die_mut=[0.001],filename='model_3d_'+str(n))



def main():
    make_figs() 

try:
    if __name__=='__main__': main()
except KeyboardInterrupt:
    traceback.print_exc()
    print 'Break!'
