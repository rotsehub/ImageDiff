"""
Few utility functions useful here and elsewhere
"""

import numpy as np
from scipy.signal import convolve2d as conv2d 
import scipy.linalg as LA
from scipy.sparse import spdiags
from scipy.sparse import issparse
import sys    

def get_background_level(image,npix=80,sextractor=False,ordinary=False):
    " Extimate the background level using photutils 0.3"
    

    from photutils.background import Background2D, SExtractorBackground
    from photutils import SigmaClip

    if sextractor:
        sigma_clip=SigmaClip(sigma=2.)
        bkg=SExtractorBackground(sigma_clip)
        bkg_value=bkg.calc_background(image)
    if ordinary:
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
    if H is not None: # need to correct this
        print "Adding regularization"
        #- first normalize: 
        iCov=iCov+H
    print "iCov Condition number",np.linalg.cond(iCov) 
    
    xsol=np.linalg.solve(iCov,y)

    w, v=LA.eigh(iCov)

    maxval=np.max(w)
    threshold=maxval*sys.float_info.epsilon
    wscaled=np.zeros_like(w)

    minval=np.sqrt(maxval)*threshold #- using sqrt
    replace=minval
    wscaled[:]=np.where((w>minval),w,replace*np.ones_like(w))
    nw=len(w)
    wdiag=spdiags(wscaled,0,nw,nw)
    
    #- construct matrix from this and eigen vectors
    sqrt_iCov=v.dot(wdiag.dot(v.T))
    
    #- get the norm vector and error
    norm_vector=np.sum(sqrt_iCov,axis=1)
    dyy=1/norm_vector

    #- resolution
    R = np.outer(norm_vector**(-1), np.ones(norm_vector.size)) * sqrt_iCov
    
    #- convolve flux so as to decorrelate the errors
    yout=R.dot(xsol) #- This removes the ringing and preserves the same resolution as data.

    #return yout#,dyy,R
    return xsol

