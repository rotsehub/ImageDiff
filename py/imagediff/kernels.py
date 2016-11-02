"""
Different Kernels
"""
import numpy as np
from scipy import special as sp
import scipy.special.orthogonal as spo

def gaussian(x,sigma=1.0):
    """
    returns the Gaussian value at a point u
    """
    gauss=1/(np.sqrt(2*np.pi*sigma))*np.exp((-x**2)/(2*sigma**2)) #Normal
    return gauss

def gauss_hermite(u,sigma=1.,m=0):
    gaus_herm=gaussian(u,sigma)*spo.eval_hermite(m,u/sigma)

    return gaus_herm

class Kernels:

    def __init__(self,name,pixelsize,sigma=None,orders=None):
        self.name=name
        self.pixelsize=pixelsize
        self.orders=orders
        self.sigma=sigma

    
class Gauss_Poly(Kernels):

    def __init__(self,name=None,pixelsize=None,sigma=None,orders=None):
        if name is None:
            name="gauss_poly"
        if pixelsize is None:
            pixelsize=9
        if sigma is None:
            sigma=[1.,1.5,3.5]
        if orders is None:
            orders=[4,3,2]

        Kernels.__init__(self,name,pixelsize,sigma,orders)

    def this_kernel_values(self,u,v,sigma,orders):

        dim=[]
        for ii in range(len(orders)):  # an order 6 has 28 basis
            p=(orders[ii]+1)*(orders[ii]+2)/2
            dim.append(p)    
        kernel_dim=np.sum(dim)
        kernel_values=np.zeros(kernel_dim)

        nn=0
        for ii in range(len(sigma)):
            u_gkernel=gaussian(u,sigma[ii])
            v_gkernel=gaussian(v,sigma[ii])

            for p in range(orders[ii]+1):
                for q in range(orders[ii]+1-p):
                    kernel_values[nn]=np.outer(v_gkernel,u_gkernel) * u**p * v**q
                    nn+=1
        self.one_kernel_values=kernel_values
        return kernel_values
        
    def all_kernels(self):

        uarr=np.linspace(int(self.pixelsize/2)*-1,int(self.pixelsize/2),self.pixelsize)
        varr=np.linspace(int(self.pixelsize/2)*-1,int(self.pixelsize/2),self.pixelsize)

        sigma=self.sigma
        orders=self.orders

        dim=[]
        for ii in range(len(orders)):
            p=(orders[ii]+1)*(orders[ii]+2)/2
            dim.append(p)
        kernel_dim=np.sum(dim)
        self.kernel_dim=kernel_dim

        finalkernels=np.zeros((len(uarr),len(varr),kernel_dim)) #basis kernels 

        for ii in range(len(uarr)):
            for jj in range(len(varr)):
                finalkernels[ii,jj,:]=self.this_kernel_values(uarr[ii],varr[jj],sigma,orders)

        #Normalize the basis kernels
        for kk in range(kernel_dim):
            finalkernels[:,:,kk]=finalkernels[:,:,kk].clip(0)
            finalkernels[:,:,kk]/=np.sum(finalkernels[:,:,kk])

        self.kernel_values=finalkernels
        return finalkernels


class Gauss_Hermite(Kernels):

    def __init__(self,name=None,pixelsize=None,sigma=None,orders=None):
        if name is None:
            name="gauss_hermite"
        if pixelsize is None:
            pixelsize=9
        if sigma is None:
            sigma=[0.7,1.5,3]
        if orders is None:
            orders=4
        Kernels.__init__(self,name,pixelsize,sigma,orders)
        
    
    def this_kernel_values(self,u,v,sigma,orders):

        dim=[]    
        kernel_dim=((orders+1)*(orders+2)/2)*len(sigma)
        kernel_values=np.zeros(kernel_dim)

        nn=0

        for ii in range(len(sigma)):
            for p in range(orders+1):
                for q in range(orders+1-p):
                    u_ghkernel=gauss_hermite(u,sigma[ii],m=p)
                    v_ghkernel=gauss_hermite(v,sigma[ii],m=q)
                    kernel_values[nn]=np.outer(v_ghkernel,u_ghkernel)
                    nn+=1
        self.one_kernel_values=kernel_values
        return kernel_values


    def all_kernels(self):

        uarr=np.linspace(int(self.pixelsize/2)*-1,int(self.pixelsize/2),self.pixelsize)
        varr=np.linspace(int(self.pixelsize/2)*-1,int(self.pixelsize/2),self.pixelsize)

        sigma=self.sigma
        m=self.orders

        kernel_dim=((m+1)*(m+2)/2)*3 #- (4+1)*(4+2)/2
        self.kernel_dim=kernel_dim
        finalkernels=np.zeros((len(uarr),len(varr),kernel_dim))

        for ii in range(len(uarr)):
            for jj in range(len(varr)):
                finalkernels[ii,jj,:]=self.this_kernel_values(uarr[ii],varr[jj],sigma,orders=m)

        for kk in range(kernel_dim):
            finalkernels[:,:,kk]=finalkernels[:,:,kk].clip(0)
            finalkernels[:,:,kk]/=np.sum(finalkernels[:,:,kk]) 

        self.kernel_values=finalkernels
        return finalkernels

class Delta(Kernels):

    def __init__(self,name=None,pixelsize=None):
        if name is None:
            name="delta"
        if pixelsize is None:
            pixelsize=9

        Kernels.__init__(self,name,pixelsize)

    def all_kernels(self):

        uarr=np.linspace(int(self.pixelsize)*-1,int(self.pixelsize),self.pixelsize)
        varr=np.linspace(int(self.pixelsize)*-1,int(self.pixelsize),self.pixelsize)


        nu=len(uarr)
        nv=len(varr)
        kernel_dim=nu*nv

        finalkernels=np.zeros((nu,nv,kernel_dim))
        p=0
        for ii in range(len(uarr)):
            q=0
            for jj in range(len(varr)):
                if ii==p and jj==q:
                    finalkernels[ii,jj,ii+jj*nu]=1.
                 
                q+=1
            p+=1
        self.kernel_values=finalkernels.clip(0)
        return finalkernels.clip(0)


    def make_R_matrix(self,dim):

        nx=dim
        ny=dim
        npix=nx*ny
        new_mat=np.zeros((npix,npix)) # add 2 rows top and bottom
        for ii in xrange(2,npix-2):
            new_mat[ii,ii-2:ii+2]=1.
            new_mat[ii,ii]-=5.

        new_mat[0,0]=-4.
        new_mat[0,1]=1.
        new_mat[0,2]=1.
        new_mat[1,0]=1.
        new_mat[1,1]=-4
        new_mat[1,2]=1.
        new_mat[1,3]=1.
    
        new_mat[npix-2,npix-2]=-4
        new_mat[npix-2,npix-1]=1.
        new_mat[npix-2,npix-3]=1.
        new_mat[npix-2,npix-4]=1.
        new_mat[npix-1,npix-1]=-4.
        new_mat[npix-1,npix-2]=1.
        new_mat[npix-1,npix-3]=1.
        new_mat[npix-3,npix-1]=1.
    
        R=new_mat
        return R

