#!/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
#import KDEpy
from scipy.stats import norm
#from scipy.stats import multivariate_normal as norm3D
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import multivariate_normal as norm3
import sys
from scipy.stats import norm



def minmax(d1,d2,smp):
    max1=np.max(d1)
    max2=np.max(d2)
    max3=np.max(smp)
    min1=np.min(d1)
    min2=np.min(d2)
    min3=np.min(smp)
    margin=0.05
    max_edge=max([max1,max2,max3])+margin
    min_edge=min([min1,min2,min3])-margin
    RANGE=(min_edge,max_edge)
    return RANGE
def minmax6(D1,D2,S):
    #Give it a numpy array
    ind=[2,3,4,6,7,8]
    R=[]
    for i in ind:
        r=minmax(D1[:,i],D2[:,i],S[:,i])
        R.append(r)
    return R


def makehists1D(RootFile,Name,P1,P2,S,M1,M2):
    plt.style.use('seaborn-deep')
    R=minmax6(P1,P2,S)
    ind=[2,3,4,6,7,8]
    Title=["Theta12","Theta34","Theta23","r2","r3","r4"]
    XLabel=[r'$\vartheta_{12}$',r'$\vartheta_{34}$',r'$\vartheta_{23}$',r'$\frac{R2}{R1}$',r'$\frac{R3}{R1}$',r'$\frac{R4}{R1}$']
    YLabel="Counts"
    for i in range(6): 
        n=ind[i]
        plt.cla()
        fig, ax = plt.subplots()
        #ax.hist([P1[:,n],P2[:,n]],range=R[i],bins=30,label=[M1,M2],color=['b','g'])
        
        ax.hist(P1[:,n],range=R[i],bins=30,color='b',label=M1,alpha=0.5,edgecolor='k')
        ax.hist(P2[:,n],range=R[i],bins=30,color='g',label=M2,alpha=0.5,edgecolor='k')
        legend = ax.legend(loc='best', shadow=True, fontsize='large')
        #legend.get_frame().set_facecolor('C1')
        ax.set_title(Name+" "+XLabel[i])
        ylabel=ax.set_ylabel(YLabel)
        xlabel=ax.set_xlabel(XLabel[i])
        saveName=Name.replace(' ','_')
        savepath=RootFile+saveName+"_"+M1+"_"+M2+"/"+Title[i]+".png"
        fig.savefig(savepath)
        plt.close(fig)
        #plt.show()
        



#Basically the same function as in 3D, but now NBins is an Array
def BayesSimple6(POPULATION1,POPULATION2,SAMPLE,NBINS,RANGE):
    T12=POPULATION1[:,2]
    T34=POPULATION1[:,3]
    T23=POPULATION1[:,4]
    T2=POPULATION1[:,6]
    T3=POPULATION1[:,7]
    T4=POPULATION1[:,8]
    normP1=T23.size
    #print(normP1)
    O12=POPULATION2[:,2]
    O34=POPULATION2[:,3]
    O23=POPULATION2[:,4]
    O2=POPULATION2[:,6]
    O3=POPULATION2[:,7]
    O4=POPULATION2[:,8]
    normP2=O23.size
    t12=SAMPLE[:,2]
    t34=SAMPLE[:,3]
    t23=SAMPLE[:,4]
    t2=SAMPLE[:,6]
    t3=SAMPLE[:,7]
    t4=SAMPLE[:,8]
    normS=t23.size
    #Bin It
    HP1,EP1=np.histogramdd((T12,T34,T23,T2,T3,T4),bins=NBINS, range=RANGE)
    HP2,EP2=np.histogramdd((O12,O34,O23,O2,O3,O4),bins=NBINS, range=RANGE)
    HS,ES=np.histogramdd((t12,t34,t23,t2,t3,t4),bins=NBINS, range=RANGE)
    fact1=normP1/normS
    fact2=normP1/normP2
    HS=HS*fact1
    HP2=HP2*fact2
    #print(np.sum(HS1))
    Log1=np.zeros(HP1.shape,dtype=np.float64)
    Log2=np.zeros(HP2.shape,dtype=np.float64)
    #Log1G=np.zeros(HP1.shape,dtype=np.float64)
    #Log1P=np.zeros(HP1.shape,dtype=np.float64)
    #Log2G=np.zeros(HP2.shape,dtype=np.float64)
    #Log2P=np.zeros(HP2.shape,dtype=np.float64) 
    Mask1=(HP1>0).astype(int)
    Mask2=(HP2>0).astype(int)
    #MaskP1_NoSter=((HP1>0).astype(int))*((HS<25).astype(int))
    #MaskP2_NoSter=((HP2>0).astype(int))*((HS<25).astype(int))
    i1=np.where(Mask1==1)
    i2=np.where(Mask2==1)
    Log1[i1]=(-HP1[i1])+HS[i1]*np.log(HP1[i1])
    #-np.log(sp.special.factorial(HS[i1]))
    Log2[i2]=(-HP2[i2])+HS[i2]*np.log(HP2[i2])
    #-np.log(sp.special.factorial(HS[i2]))
    #-------------------------------------------------------#
    #Sterling Approx for HS>40, cause why not. Numpy is fast
    #Log1Ster=np.zeros(HP1.shape,dtype=np.float64)
    #Log2Ster=np.zeros(HP2.shape,dtype=np.float64)
    #MaskSter1=((HP1>0).astype(int))*((HS>24).astype(int))
    #MaskSter2=((HP2>0).astype(int))*((HS>24).astype(int))
    #iS1=np.where(MaskSter1==1)
    #iS2=np.where(MaskSter2==1)
    #Log1Ster[iS1]=(-HP1[iS1])+HS[iS1]*np.log(HP1[iS1])-(HS[iS1]*np.log(HS[iS1])-HS[iS1] +0.5*np.log(2*np.pi*HS[iS1]))
    #Log2Ster[iS2]=(-HP2[iS2])+HS[iS2]*np.log(HP2[iS2])-(HS[iS2]*np.log(HS[iS2])-HS[iS2] +0.5*np.log(2*np.pi*HS[iS2]))
    #Log1Ster=Log1Ster
    #Log2Ster=Log2Ster## a general mask that just rules out HP values for either 
    #population for which a Poisson PDF cannot be evaluated
    #---------------------------------------------#
    #Log1=Log1P+Log1Ster
    #Log2=Log2P+Log2Ster
    Log1R=np.zeros(HP1.shape,dtype=np.float64)
    Log2R=np.zeros(HP2.shape,dtype=np.float64)
    Log1R=Log1*Mask1*Mask2
    Log2R=Log2*Mask1*Mask2
    mask1=(Log1R!=0).astype(int)
    mask2=(Log2R!=0).astype(int)
    #print(np.sum(Mask1),np.sum(Mask2),np.sum(Mask1*Mask2))
    EXP_SUM=np.sum(Log1R-Log2R,dtype=np.float64)
    N=np.sum(Mask1*Mask2)
    a=Log1
    b=Log2
    #print(is_num)&
    #np.isnanEXP_Not_NaN
    #print(np.where(np.isnan(EXP)==False))
    SimpleBayes=EXP_SUM
    return a,b,EXP_SUM,HP1.flatten(),HP2.flatten(),HS.flatten(),N




def LongBayes(POPULATION1,POPULATION2,SAMPLE,NBINS,NUMCUTS,DIM):
    ###Make your cut dimiension bin # divisible by the number of cuts
    RANGE=minmax6(POPULATION1,POPULATION2,SAMPLE)
    B=0
    N=0
    NBINS[DIM]=int(list(NBINS)[DIM]/NUMCUTS)
    #print(NBINS)
    r=RANGE[DIM]
    Delta=(r[1]-r[0])/NUMCUTS
    for i in range(NUMCUTS):
        RANGE[DIM]=(r[0]+i*Delta,r[0]+(i+1)*Delta)
        a,b,EXP_SUM,HP1,HP2,HS,num=BayesSimple6(POPULATION1,POPULATION2,SAMPLE,NBINS,RANGE) 
        N=N+num
        B=B+EXP_SUM
    return B,N,B/N

#def LongLongBayes(POPULATION1,POPULATION2,SAMPLE,NBINS,NUMCUTS,SUBCUTS,DIM,SUBDIM)
