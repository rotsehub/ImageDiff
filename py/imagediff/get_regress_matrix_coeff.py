#! /usr/bin/env python

"""
   Find the regression matrix coefficients from 
   minimizing by least square method
   Work Here follows: Becker et. al.: http://arxiv.org/pdf/1202.2902v1.pdf
   Section 2.1
"""

from astropy.convolution import convolve_fft as conv2d_fft
from scipy.signal import convolve2d as conv2d 
import scipy.linalg
import numpy as np
from scipy.sparse import spdiags,issparse


def get_background_iqd(image):

    """
    uses interquartile difference to evaluate the background etc
    returns median, stddev
    """
    img_data=np.sort(image,axis=None)

    lim1=np.percentile(image,(25,75))
    stdim=np.std(img_data[(img_data>lim1[0]) & (img_data<lim1[1])])
    medim=np.median(img_data)
    print "stdim", stdim, "median", medim
    den=5.3959180 ## 4*(gauss_cvf(0.25)-gauss_cvf(0.75)) from idl code

    n=img_data.shape[0]

    #- for stddev:
    m=np.mod(n,4)
    r=n/4
    def del_val(m,x):
        ix=range(x.shape[0])
        if m==0:
           delv=2*(x[ix[n-r-1]]+x[ix[n-r]]-x[ix[r]]-x[ix[r-1]])
        if m==1:
           delv=3*(x[ix[n-r-1]]-x[ix[r]])+(x[ix[n-r]]-x[ix[r-1]])
        if m==2:
           delv=4*(x[ix[n-r-1]]-x[ix[r]])
        if delv==3:
           delv=(x[ix[n-r-2]]-x[ix[r+1]])+3*(x[ix[n-r-1]]-x[ix[r]])
        return delv

    delvim=del_val(m,img_data)

    std_dev=delvim/den
    #- for median:
    mm=np.mod(n,2)
    rr=n/2
    def dell_val(mm,x):
        ix=range(x.shape[0])
        if mm==0:
            q=2*(x[ix[rr-1]]+x[ix[rr]])/4
        if mm==1:
            q=4*x[ix[rr]]/4
        return q/4

    median=del_val(m,img_data)/4
    return median,std_dev

def get_background_level(image,nbins=100,niter=3):
    from scipy.ndimage import filters
    #nx=image.shape[0]
    #ny=image.shape[1]
    #if nbins > nx:
    #    nbins=nx
    #sky_background=np.zeros((nx,ny))
    #bkg=image #- initialize as image
    #- iterate this three times 
    #for jj in range(niter):

#        for ii in range(nx): #column wise median filtering
#            sky_bkg=filters.median_filter(bkg[ii,:],nbins*(1+jj))
#            sky_background[ii,:]=sky_bkg.clip(0,np.percentile(sky_bkg,25))
#        bkg=sky_background
#
#    return bkg
    pix=image.ravel()
    #bkg=filters.median_filter(pix,100)
    sky_bkg=filters.median_filter(pix,50)
    
    return sky_bkg.reshape(image.shape)       
    
def get_convolve_kernel(imgarray,kernel):
    """
    imgarray: [nx,ny] 2d array like image
    kernel: [n,n,N] eg. [21,21,49]
    mode: scipy.special.convolve2d mode
          'same' gives same dimension as imgarray
    """
    convec=np.zeros((kernel.shape[2],imgarray.size))
    for i in range(kernel.shape[2]):
        conv=conv2d(imgarray,kernel[:,:,i],mode='same').ravel()
        convec[i,:]=conv
    return convec
      

def get_per_pixel_variance(imgarray,readnoise=1.,randseed=0):

    """
    sigma in section 2.1 Becker et. al.
    """
    from scipy.sparse import spdiags

    np.random.seed(randseed)
    #imgarray=imgarray.clip(0.1)
    #readnoise=np.random.normal(scale=readnoise,size=imgarray.shape)
    #pix=np.random.poisson(imgarray)+readnoise
    pix=imgarray#+readnoise**2
    ivar=1./(pix.clip(0)+readnoise**2)
    #W=spdiags(ivar.ravel(),0,ivar.shape[0],ivar.shape[1])
    #Wt=W.toarray()
    #Wt=ivar
    #variance=1/Wt
    #return W.toarray() ##weight matrix (1/sigma**2)
    variance=np.linalg.inv(ivar)
    print "variance_size",variance.shape
    return variance  

def get_covariance(imgarray):
    """Noise matrix taking independent noise"""
    covariance=np.zeros_like(imgarray)
    for ii in range(imgarray.shape[0]):
        covariance[ii,ii]=((np.std(imgarray[ii,:]))**2+(np.std(imgarray[:,ii]))**2)**0.5
    return covariance

    
def compute_matrix(C=None,S=None,variance=None):

    """ Solving for a im b=Ma same representation
    C- convolution kernel x R (n,n,N)
       N - basis dimension
    returns b vector and M (N,N) matrix 
    """
    nbasis=C.shape[2]
    b=np.zeros(nbasis)
    b_old=np.zeros(nbasis)
    M=np.zeros((nbasis,nbasis))
    M_old=np.zeros((nbasis,nbasis))
#    for ii in range(C.shape[2]):
#        ss=0.
#        for jj in range(C.shape[0]): # x
#            for kk in range (C.shape[1]): # y
#                p=((C[jj,kk,ii]*S[jj,kk])/variance[jj,kk])
#                ss+=p
#        b[ii]=ss
#    print 'b:', b
#    return b
    
# Alternatively, even better
#    b1=np.zeros(nbasis)
#    for ii in range(nbasis):
#        b[ii]=np.ndarray.sum(np.ndarray.sum(C[:,:,ii]*S/variance,axis=1),axis=0)
    if variance is None:
       variance=np.ones(S.shape) 

    #for ii in range(nbasis):
    #    b[ii]=(np.matmul(C[:,:,ii],S)/variance).sum(axis=1).sum(axis=0)
    #    for jj in range(nbasis):
    #        M[ii,jj]=(np.matmul(C[:,:,ii],C[:,:,jj])/variance).sum(axis=1).sum(axis=0)
   # k=np.where(variance==0.0)
   # print k
    for ii in range(nbasis):
         b[ii]=C[:,:,ii].T.dot(np.linalg.inv(variance)).dot(S).sum(axis=1).sum(axis=0)
         for jj in range(nbasis):
             M[ii,jj]=C[:,:,ii].T.dot(np.linalg.inv(variance)).dot(C[:,:,jj]).sum(axis=1).sum(axis=0)
    return b, M

def solve_leastsq(M,b):
    """
    given b, M matrix, solves for least square solutions (a) to 
    b=Ma equation by minimizing the Euclidean 2d norm ||b-Ma||^2
    """
    a=np.linalg.lstsq(M,b)
    return a[0]

def solve_coeff(convolve,variance,image,H=None):
    #theta=sum_k:v_l^k(t)
    #Model_l=sum_t: theta_l,t *c_t

    Matrix=np.zeros((convolve.shape[2],convolve.shape[2]))
    Vector=np.zeros((convolve.shape[2]))
    for ii in range(convolve.shape[2]):

        part1=convolve[:,:,ii].T.dot(np.linalg.inv(variance)).dot(image)
        Vector[ii]=part1.sum()
        for jj in range(convolve.shape[2]):
 
            part2=convolve[:,:,ii].T.dot(np.linalg.inv(variance)).dot(convolve[:,:,jj])
            Matrix[ii,jj]=part2.sum()

    ## Apply unitarity
    #totvec=np.zeros(allk.shape[2]) 
    #for jj in range(allk.shape[2]):
    #    totvec[jj]=np.sum(allk[:,:,jj])
    
    #for jj in xrange(1,allk.shape[2]):
    #    Matrix[:,jj]=Matrix[:,jj]-Matrix[:,0]*totvec[jj]/totvec[0]
    #    Matrix[jj,:]=Matrix[jj,:]-Matrix[0,:]*totvec[jj]/totvec[0]
    #    Vector[jj]=Vector[jj]/totvec[0]
    if H is not None:
        lamb=1
        Matrix+=lamb*H
    print "matrix_ cond number", np.linalg.cond(Matrix)
    coeff=np.dot(np.linalg.inv(Matrix),Vector)
    print "Matrix", Matrix
    print "Solution",Vector
    lsq=np.linalg.lstsq(Matrix,Vector)
    return coeff,lsq[0]

def norm(x):
    return np.sqrt(np.dot(x,x))

def solve_for_coeff(A,b,W=None,H=None):

    """
    A: Matrix: (nobs,nvar)
    b: vector: (science pix) (nvar)
    W: wt vector: (nvar)
    """

    if W is None:
       W=np.ones(A.shape[1])
    
    #- Inverse covariance matrix
    iCov=A.T.dot(W.dot(A))
    y=A.T.dot(W.dot(b))
    print A.shape,b.shape,W.shape
    if H is not None: # need to correct this
        print "Adding regularization"
        #- first normalize: 
        # pix should is background subtracted
        iCov=iCov+H
    print "iCov Condition number",np.linalg.cond(iCov) 
    #- Vector 
    
    #y=A.T.dot(W.dot(pix))
    # or y=A.T.dot( W*pix ) 
    #- Solve
    xsol=np.linalg.lstsq(iCov,y)[0]
    #effectivetemp=A.dot(xsol[0])
    #effectivetemp=effectivetemp.reshape(nx,ny)
    #print "shape of eff",effectivetemp.shape
    return xsol

def resol_matrix(iCov):
    """ inverse Covariance matrix
    """
    import sys 

    iCov=0.5*(iCov+iCov.T)
    if issparse(iCov):
        iCov=iCov.toarray()

    w,v=scipy.linalg.eigh(iCov)
    dim=w.shape[0]
    print 'dim:',dim
    sqrt_iCov=np.zeros_like(iCov)

    #- eigen decomposition
    maxval=np.max(w) #- max eigen value
    wscaled=np.zeros_like(w)
    threshold=10.0*sys.float_info.epsilon
    minval=np.sqrt(maxval)*threshold #- 10 times machine precision ~1e-15
    replace=minval
    wscaled[:]=np.where((w>minval),np.sqrt(w),replace*np.ones_like(w))
    wdiag=spdiags(wscaled,0,dim,dim)
    sqrt_iCov=v.dot( wdiag.dot(v.T) ) 
    
    norm_vector=np.sum(sqrt_iCov,axis=1)
    R=np.outer(norm_vector**(-1),np.ones(norm_vector.size))*sqrt_iCov
    ivar=norm_vector**2
    print "R-shape:",R.shape, ivar.shape
    return R,ivar
    
def chi_sq(diffimg,img_variance):
    return np.sum(diffimg**2/img_variance)

def make_pca(X):
    """ returns projection matrix (with importance decending) and variance
        X should be row flattened array
    """
    
    num_data,dim=X.shape

    mean_X=X.mean(axis=0)
    for i in range(num_data):
        X[i]-=np.mean(X)
    if dim>100: 
        print "PCA -Compact trick"
        M=np.dot(X,X.T) # covariance
        e,EV=scipy.linalg.eigh(M)
        k=np.where(e < 0)
        if k[0].shape[0]>0:
           print "negative eigen values!!!", k
        tmp=np.dot(X.T,EV).T #compact trick??
        V=tmp[::-1] # reverse since last eigen vectors are desired
        S=np.sqrt(e)[::-1] # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:,i]/=S  # normalization

    else:
        print "PCS- SVD used"
        U,S,V=scipy.linalg.svd(X)
        V=V[:num_data] # retruning first num_data

    # return the projection matrix, Variance and mean
      
    return V,S,mean_X
   
def get_PCA_basis(imlist):
    """
       imlist: is 3d: (nx,ny,nim)
    """
    nx,ny=imlist.shape[0:2] #image size
    n_im=imlist.shape[2] 
    
    # make a matrix to store all flat images
    immatrix=np.array([imlist[:,:,i].flatten() for i in range(n_im)],'f')            
    
    # now work for PCA
    V,S,imean=make_pca(immatrix)
    
    # get all the modes
    #modes=V
    modes=np.zeros((nx,ny,n_im))
    for i in range(n_im):

        modes[:,:,i]=V[i].reshape(nx,ny)
        
    return modes
