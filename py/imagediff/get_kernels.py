#!/usr/bin/env python
import numpy as np
from scipy import special as sp
import scipy.special.orthogonal as spo

def Gaussian(x,sigma=1.0):
    """
    returns the Gaussian value at a point u
    """
    #gauss=1/np.sqrt(2*np.pi*sigma**2)*np.exp((-u**2)/(2*sigma**2))
    gauss=1/(np.sqrt(2*np.pi*sigma))*np.exp((-x**2)/(2*sigma**2)) #Normal
    return gauss


#def Gauss_Hermite(u,sigma=1,m=0):
#    gauss_hermite=Gaussian(u,sigma)*spo.eval_hermite(m,u/sigma)

#    return gauss_hermite

#def Gauss_Hermite_integral(x,m=0,sigma=1.):
#    """ Normalized"""
#    hermite_norm=list()
#    for i in range(m+1):
#        hermite_norm.append(sp.hermitenorm(i))
#
#    dx=x-0.5
#    u=np.concatenate((dx,dx[-1:]+0.5))/sigma
#
#    if m>0:
#        y=hermite_norm[m-1](x)*np.exp(-0.5*u**2)/np.sqrt(2.*np.pi)
#        return (y[1:]-y[0:-1])
#    else:
#        y=sp.erf(u/np.sqrt(2.))
#        return 0.5*(y[1:]-y[0:-1])

#def xy_gauss_hermite(x,y,m=0,n=0,sigma=1):
#
#    # Core
#    xgh=Gauss_Hermite_integral(x,m,sigma=sigma)
#    ygh=Gauss_Hermite_intefral(y,n,sigma=sigma)
#    core=np.outer(ygh,xgh)
       
"""
def find_kernel_values(u,v,sigma=1.,m=0,n=0,order=0):

    
    #order: 0<=order<=i+j
    

    u_ghkernel=Gauss_Hermite(u,sigma,m)
    v_ghkernel=Gauss_Hermite(v,sigma,n)

    ghkernel=u_ghkernel*v_ghkernel

    kernel_dim=(order+1)*(order+2)/2

    kernel_values=np.zeros(kernel_dim) # an order 6 has 28 basis

    for i in range(kernel_dim):
        
        p1=0
        for p in range(order+1):
            q1=0
            for q in range(order+1-p):
                k=ghkernel*u**p*v**q
                q1+=k
             #   print 'q1:', q1
            p1+=q1
            #print 'p1:',p1
        kernel_values[i]+=p1 
    return kernel_values
            
           
def make_kernels(uarr,varr,sigma=1.,m=0,n=0,order=0):
     
     #uarr: array of u_i
     #varr: array of v_i       
     
     kernel_dim=(order+1)*(order+2)/2
     #for ii in range(m[0]):
     kernels=np.zeros((len(uarr),len(varr),kernel_dim))
     for ii in range(len(uarr)):
         for jj in range(len(varr)):
             for kk in range(kernel_dim):
                 kernels[ii,jj,kk]=find_kernel_values(uarr[ii],varr[jj],sigma)

     return kernels
"""      

def make_one_point_kernels(u,v,sigma=None,N=3,orders=None):


    if sigma is None:
        sigma=[0.7,1.5,3.0]
    if orders is None:
        orders=[6,4,2]
       
    dim=[]

    for ii in range(len(orders)):  # an order 6 has 28 basis
        p=(orders[ii]+1)*(orders[ii]+2)/2
        dim.append(p)
    
    kernel_dim=np.sum(dim)

    kernel_values=np.zeros(kernel_dim)

    #multiply the Gaussian by polynomial terms 
    nn=0
    for ii in range(N):
        u_gkernel=Gaussian(u,sigma[ii])
        v_gkernel=Gaussian(v,sigma[ii])
        kernel_val=np.outer(v_gkernel,u_gkernel)

        for p in range(orders[ii]+1):
            for q in range(orders[ii]+1-p):
                kernel_values[nn]=kernel_val*u**p*v**q
             #   print nn
                nn+=1
        #m+=(order[ii]+1)*(order[ii]+2)/2
    return kernel_values    
    
def total_kernels(uarr,varr,sigma=None,N=3,orders=None):
    """
    N: No. of Gaussian
    sigma: list of sigmas for each Gaussian
    orders: degree for polynomial(i+j) for each case
    """
  
    if sigma is None:
        sigma=[0.7,1.5,3.0]
    if orders is None:
        orders=[6,4,2]

    dim=[]
    for ii in range(len(orders)):
        p=(orders[ii]+1)*(orders[ii]+2)/2
        dim.append(p)
    kernel_dim=np.sum(dim)

    finalkernels=np.zeros((len(uarr),len(varr),kernel_dim)) #basis kernels

    for ii in range(len(uarr)):
        for jj in range(len(varr)):
            finalkernels[ii,jj,:]=make_one_point_kernels(uarr[ii],varr[jj],sigma=sigma,N=N,orders=orders)
    #Normalize the basis kernels
    for kk in range(kernel_dim):
        finalkernels[:,:,kk]=finalkernels[:,:,kk].clip(0.0)
        finalkernels[:,:,kk]/=np.sum(finalkernels[:,:,kk]) 
        #finalkernels[:,:,kk].clip(0.0)       
    return finalkernels
