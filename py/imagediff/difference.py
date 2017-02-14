import numpy as np
from imagediff import kernels, util

def do_subtraction(image,template,cutoff=1.0,radius=10,boundary=None,maskreg=None, kerneltype=None,method=None,sqrt=False,rdnoise=False):

    """
    image: fits object
    template: fits object
    """

    if kerneltype is None: #- Gaussian poly as default
        kerneltype="gauss_poly"

    #- Check for kerneltype

    if kerneltype not in ["gauss_poly","gauss_hermite","delta"]:
        raise ValueError("Not a valid kernel type. Give a valid kernel type.")

    pixelsize = 11

    #- instantiate the base Kernel
    Kern=kernels.Kernels(kerneltype, pixelsize)

    #- get all kernels for this type

    if Kern.name == "gauss_poly" :   
        KK=kernels.Gauss_Poly(Kern) #- using default sigmas and orders
        allk=KK.all_kernels()
    if Kern.name == "gauss_hermite":
        KK=kernels.Gauss_Hermite(Kern)
        allk=KK.all_kernels()
    if Kern.name == "delta":
        KK=kernels.Delta(Kern)
        allk=KK.all_kernels()
        #H=KK.make_R_matrix

    print "using kernel", kerneltype

    mask=np.ones_like(image[0].data)

    print "mask shape", mask.shape
    if maskreg is not None:
        print "Mask reg.", maskreg
        mxlo,mxhi,mylo,myhi=map(int, maskreg.split(','))
        mask[mxlo:mxhi,mylo:myhi]=0

    if boundary is not None:
        xlo,xhi,ylo,yhi=map(int, boundary.split(','))
    else:
        xlo,xhi,ylo,yhi=0,image[0].data.shape[0],0,image[0].data.shape[1]
   
    #- image
    image_data=image[0].data[xlo:xhi,ylo:yhi]#*mask[xlo:xhi,ylo:yhi]    
    sky_image=util.get_background_level(image_data)
    Zs=image_data-sky_image 
    imghdr=image[0].header    
    if imghdr["PC001001"] < 0:  #- Coordinate rotation matrix opposite!
        Zs=np.rot90(np.rot90(Zs))
        print "rotating image"  
       
    #- template
    template_data=template[0].data[xlo:xhi,ylo:yhi]#*mask[xlo:xhi,ylo:yhi]
    sky_template=util.get_background_level(template_data)
    Zt=template_data-sky_template
    temphdr=template[0].header
    if temphdr["PC001001"] < 0:  #- Coordinate rotation matrix opposite!
        Zt=np.rot90(np.rot90(Zt))
        print "rotating template"

    #- Get needed header parameters to propagate from ROTSE image file
    satcnts_t=temphdr["SATCNTS"] 
    satcnts_i=imghdr["SATCNTS"] 
    
    #- mask all pixels higher than cutoff of saturation level
    print "SatCnts_i", satcnts_i
    k_t=np.where(Zt>satcnts_t*cutoff)
    print "SatCnts_t", satcnts_t

    for ii in range(k_t[0].shape[0]):
        Zt[k_t[0][ii]-radius:k_t[0][ii]+radius,k_t[1][ii]-radius:k_t[1][ii]+radius]=1.0e-30
        Zs[k_t[0][ii]-radius:k_t[0][ii]+radius,k_t[1][ii]-radius:k_t[1][ii]+radius]=1.0e-30
    k_picked=np.where(Zt==1.0e-30)[0]    

    print "No. of pixels above cutoff", k_t[0].shape[0]

    #- Add noise to the masked regions:
    #print "count", Zt[Zt==1.0e-30]
    rmsi=np.std(Zs[(Zs>np.percentile(Zs,5)) & (Zs< np.percentile(Zs,95))])
    rmst=np.std(Zt[(Zt>np.percentile(Zt,5)) & (Zt < np.percentile(Zt,95))])
    Zt[Zt==1.0e-30]=np.random.normal(0,rmst,size=k_picked.shape)
    Zs[Zs==1.0e-30]=np.random.normal(0,rmsi,size=k_picked.shape)

    if rdnoise:
        readnoise_t=temphdr["BSTDDEV"]
        readnoise_i=imghdr["BSTDDEV"]
    else:
        #- assume a sparse noisy image. derive from the pixel counts. This is too large!
        #readnoise_i=np.median(Zs)-np.percentile(Zs,15.865)
        #readnoise_t=np.median(Zt)-np.percentile(Zt,15.865)
        readnoise_i=5.
        readnoise_t=5.

    print "Rdnoise image:", readnoise_i
    print "Rdnoise template:", readnoise_t
    exptime_t=temphdr["EXPTIME"]
    exptime_i=imghdr["EXPTIME"]
     
    sc_variance=np.abs(Zs)+readnoise_i**2 # Zs.clip(0) ???
    temp_variance=np.abs(Zt)+readnoise_t**2   
    #tot_variance=sc_variance+temp_variance    

    convtemp=util.get_convolve_kernel(Zt,allk) #2D [nvec,nobs]
    #- subreg=int(sc.shape[0]/300)+1 #- dividing into subregions
    #- print "subreg",subreg
    efftemplate=np.zeros_like(Zs)
    #n=300

    if method == "PCA":

        #- use empca
        print "using pca"
        #- this requires empca package https://github.com/sbailey/empca/
        import empca
        nx,ny=Zs.shape[0:2]
        n_im=convtemp.shape[0]

        #- list convolved template as data
        data=np.array([convtemp[i,:] for i in range(convtemp.shape[0])],'f')
        var_data=data.clip(0.)+readnoise_t**2
        wt=1./var_data

        #- Add image to the data list
        all_data=np.vstack((data,Zs.ravel()))
        all_wt=np.vstack((wt,1./sc_variance.ravel()))
        if sqrt: 
            all_wt=all_wt**0.5
        model=empca.empca(all_data,all_wt,niter=15,nvec=10,smooth=0)
        efftemplate=model.eigvec.T.dot(model.coeff[-1]).reshape(nx,ny)
        #dof = all_data.size-10*nx-10*(n_im+1)
        #print "dof",dof
        #Chisq=np.sum((effdata-all_data.T)**2*all_wt.T)/dof
        #print "Chisq_pca", Chisq

    else: #- kernel based
        print "kernel based"
        A=convtemp.T
        ivar=1./sc_variance.ravel()
        from scipy.sparse import spdiags
        if sqrt:
            wt=spdiags(np.sqrt(ivar),0,ivar.size,ivar.size) 
        else:
            wt=spdiags(ivar,0,ivar.size,ivar.size)
        b=Zs.ravel()
        xsol=util.solve_for_coeff(A,b,W=wt)
        efftemplate=convtemp.T.dot(xsol).reshape(Zs.shape)
        
    #- diff image
    diffimage=(Zs-efftemplate)
    if maskreg is not None: 
        diffrms=np.std(diffimage[(diffimage>np.percentile(diffimage,5)) & (diffimage<np.percentile(diffimage,95))])
        print "diff rms",diffrms, 'rmsi',rmsi
        diffimage[mxlo:mxhi,mylo:myhi]=np.random.normal(scale=diffrms,size=(mxhi-mxlo,myhi-mylo)) 
  
    # mse estimate: 
    mse=1./(np.size(diffimage)) * diffimage.ravel().T.dot(diffimage.ravel())
    print "mse", mse

    # Quality metrics

    ii = np.where(sc_variance>0)
    dof=Zs.size+convtemp.size-convtemp.shape[0]
    print "dof",dof        
    chisq=np.sum(diffimage[ii]**2/sc_variance[ii])/dof
    print "chisq/dof", chisq
    
    #- fraction of the variance recovered.
    
    R_var=1.0-np.var(diffimage[ii])/np.var(Zs[ii])

    return diffimage, efftemplate, Zs, R_var,chisq

