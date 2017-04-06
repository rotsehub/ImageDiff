"""
Few utility functions useful here and elsewhere
"""

import numpy as np
from scipy.signal import convolve2d as conv2d 
import scipy.linalg as LA
from scipy.sparse import spdiags
from scipy.sparse import issparse
import sys    

def get_background_level(image,npix=50,sextractor=False,ordinary=False):
    " Extimate the background level using photutils 0.3"
    

    from photutils.background import Background2D, SExtractorBackground
    from photutils import SigmaClip
    if npix> image.shape[0]: 
       print "Image shape", image.shape
       npix=image.shape[0]-1
    if sextractor:
        sigma_clip=SigmaClip(sigma=2.)
        bkg=SExtractorBackground(sigma_clip)
        bkg_value=bkg.calc_background(image)
    elif ordinary:
       from scipy.ndimage import filters
       pix=image.ravel()
       bkg_value=filters.median_filter(pix,50).reshape(image.shape)
    else:
        bkg=Background2D(image,box_size=npix)
        bkg_value=bkg.background
    return bkg_value


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

def solve_for_coeff(A,b,W=None,H=None,resol=False):

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

    print "iCov Condition number",np.linalg.cond(iCov) 

    if resol: 
        R,ivar=get_resolution(iCov)
        print R.shape
        #- convolve flux so as to decorrelate the errors
        yout=R.dot(xsol) #- This removes the ringing and preserves the same resolution as data.
        return yout

    #- Rescale
    iCov=rescale(iCov)

    if H is None:
        #- add a weak prior to keep well conditioned:
        epsilon = sys.float_info.epsilon
        print "weak Prior:", epsilon
        iCov=iCov+epsilon*np.eye(iCov.shape[0])

    else: # need to correct this
        print "Adding regularization"
        iCov=iCov+H
    
    #- solve
    xsol=np.linalg.solve(iCov,y)

    return xsol

def rescale(M,threshold=None):
    """
    rescale the eigen values and vectors to ignore very low eigen values
    """

    #- Eigen Decomposition
    w,v=LA.eigh(M)
    wscaled=np.zeros_like(w)

    if threshold is None:
        threshold=sys.float_info.epsilon
        replace=0
    else:
        maxval=np.max(w)
        replace=np.sqrt(maxval)*threshold #- sqrt

    wscaled[:]=np.where((w>threshold),w,replace*np.ones_like(w))

    nw=len(w)
    wdiag=spdiags(wscaled,0,nw,nw)

    #- constrain eigen vectors
    if replace==0:
        k=np.where(wscaled==0)[0]
        v[:,k]=np.zeros_like(v[:,k])

    newM=v.dot(wdiag.dot(v.T))
    print "new Matrix Condition number", np.linalg.cond(newM)
    return newM

def get_resolution(iCov): #- This is based on "spectroperfectionism" - Adam and Bolton 2009. 
    #- This may not be needed as this gives per image resolution.
    #- symmetric, semi-positive definiteness:
    iCov=0.5*(iCov+iCov.T)
    
    threshold=sys.float_info.epsilon
    sqrt_iCov=rescale(iCov,threshold=threshold)

    #- get the norm vector and error
    norm_vector=np.sum(sqrt_iCov,axis=1)
    dyy=1/norm_vector
    ivar=1/dyy**2

    #- resolution
    R = np.outer(norm_vector**(-1), np.ones(norm_vector.size)) * sqrt_iCov
    
    return R, ivar
