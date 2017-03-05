import numpy as np


def hist(v,bins=20,removezero=True):
    Y,binedges=np.histogram(v,bins=bins)
    X = 0.5*(binedges[1:]+binedges[:-1])

    #check for zero counts - causes divide bY zero error
    if removezero:
        newX=[]; newY=[]
        for i in xrange(len(Y)): 
            if Y[i]>0:
                newX.append(X[i])
                newY.append(Y[i]) 
        X=np.array(newX,dtype=float); Y=np.array(newY,dtype=float)
    else: 
        X=np.array(X,dtype=float)
        Y=np.array(Y,dtype=float)
    return X,Y,binedges





def transform(X,Y):
    """
    Transform to 1st indirect moment
    """
    X = np.array(X,dtype=float); Y=np.array(Y,dtype=float)
    Y = Y / np.sum(Y)
    a = np.sum(X*Y)
    Y = [np.sum(Y[i:]*X[i:]) / a for i in xrange(len(Y))]
    return X,Y

def f(x,k,x0): 
    """
    Simons function to which the clone size data is fitted 
    """
    return np.exp(-(x-x0)/k)


def fit(X,Y):
    from scipy.optimize import curve_fit
    a,b = curve_fit(f,X,Y) 
    return a


def plot(X,Y,a,log=False,ax=None,cutoff=0.0,col1='k',col2='grey'):
    """
    Caled by plot_single and plot_multi, do not call this directly
    """
    import pylab as pl
    #if clear: pl.clf()
    if ax is None: ax = pl.gca()
    r = np.linspace(cutoff,max(X),50)
    fitY = f(r,*a)
    if log:
        Y=np.log10(Y)
        fitY=np.log10(fitY)
    ax.plot(r,fitY,color=col2)
    ax.scatter(X,Y,marker='x',color=col1)
    ax.set_xlabel('VAF')
    if not log:
        ax.set_ylabel('1st incomplete moment')
    else:
        ax.set_ylabel('log(1st incomplete moment)')

def eqn_s2(t,n,r):
    """
    Equation S2 from simons supplementary giving clone size distribution as a function of time and rate of stem cel loss / replacement
    """
    import math
    return (1.0/math.log(r*t)) * (math.exp(-n/(r*t)) / n)
    
    
     
def plot_single(V,col='k',cutoff=0.0,log=False,ax=None): 
    """
    @V: list of clone sizes 
    Plot a single clone size distribution
    """ 
    if len(V)<3: return
    if cutoff>0: V = np.extract(V>cutoff,V)
    X,Y,binedges = hist(V)
    X,Y= transform(X,Y)
    a = fit(X,Y)
    plot(X,Y,a,log=log,col1=col,col2=col,ax=ax);
    #ax.set_xlim((0,max(X)))
    #ax.set_ylim((0,max(Y)))


def plot_multi(reps,cutoff=0.0,log=False,col1='k',col2=(0.5,0.5,0.5)):
    """
    @reps: [counts rep1,counts rep2,...]
    Plot a the average of multiple clone size distributions with error bars 
    """
    bins = 20 
    data = np.zeros((len(reps),bins))
    for i,clones in enumerate(reps):
        if cutoff>0: clones = np.extract(clones>cutoff,clones)
        X,Y,binedges = hist(clones,bins=bins,removezero=False)
        X,Y = transform(X,Y)
        data[i] = Y 
   
    mean = np.mean(data,axis=0)
    mean = np.nan_to_num(mean)
    if np.max(mean)==0: return
    a = fit(X,mean)
     
    if log:  
        std = np.std(np.log10(data),axis=0)
        mean = np.log10(mean)
    else: 
        std = np.std(data,axis=0)

    from scipy import stats
    import pylab as pl
    plot(X,mean,a,log=log,cutoff=cutoff,col1=col1,col2=col2);
    pl.errorbar(X,mean,yerr=std,color=col1,ecolor=col1,fmt='x')


