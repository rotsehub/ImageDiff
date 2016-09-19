"""
Few utility functions useful here and elsewhere
"""

def project(x1,x2):
    x1=np.sort(x1)
    x2=np.sort(x2)
    Pr=np.zeros((len(x1),len(x2)))
    
    #- find the projection matrix to map x1 to x2 such that x1=Pr.dot(x2), 
    #- x1 and x2 related by linear interpolation
    
    for jj in range(len(x2)-1): #columns
        for ii in range(len(x1)-1): #rows

            if x1[ii]==x2[jj]:
                Pr[ii,jj]=1.
            if ((x1[ii]> x2[jj]) & (x1[ii] <= x2[jj+1])):
                dx=(x1[ii]-x2[jj])/(x2[jj+1]-x2[jj]) 
                Pr[ii,jj]=1-dx
                Pr[ii,jj+1]=dx
    #- Covariance on same end points should be 1
    if x2[-1]==x1[-1]:
        Pr[-1,-1]=1.
    return Pr

def resample(x,y,dy,xx):
    """ 
       Adopting from sbailey/specter based on Spectroperfectionism (Bolton & Schlegel 2009)

    Equations 4 through 16

    Args: x: 1d array of independent variables
          y: 1d array of values sampled at x
          dy: 1d array , errors on y
          xx: new grid for resampling (same units of x)
    """    
    n1=len(x)
    n2=len(xx)

    #- projection xx -> x
    Pr=project(x,xx)

    #- inverse error**2/ Weight matrix
    W=spdiags(1/dy**2,0,n1,n1)    


    iC=Pr.T.dot(W.dot(Pr)) #- gives inverse covariance     
    
    #- Now solve
    out=Pr.T.dot(W.dot(y))
    yy=np.linalg.solve(iC,out) #- should have the dimension of xx. Cholesky decomposition is fast??
    
    #- This has deconvolved the spectrum. Need to reconvolve back and get the errors
    # right and uncorrelated in the new grid. Follwing the Resolution matrix 
    # approach (Section 3) and diagonalize the error matrix 
    #- first solve for eigen values and eigen vectors:

    w, v=scipy.linalg.eigh(iC)
    nw=len(w)
    wdiag=spdiags(np.sqrt(w),0,nw,nw)
    
    #- construct matrix from this and eigen vectors
    sqrt_iC=v.dot(wdiag.dot(v.T))
    
    #- get the norm vector and error
    norm_vector=np.sum(sqrt_iC,axis=1)
    dyy=1/norm_vector

    #- resolution
    R = np.outer(norm_vector**(-1), np.ones(norm_vector.size)) * sqrt_iC
    
    #- convolve flux so as to decorrelate the errors
    yout=R.dot(yy) #- This removes the ringing and preserves the same resolution as data.

    return yout,dyy,R

