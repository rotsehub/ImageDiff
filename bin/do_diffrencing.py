#! /usr/bin/env python

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from imagediff import kernels, util
import numpy as np
import astropy.io.fits as fits
import argparse

def main():
    parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image",type=str, help="input image to plot")
    parser.add_argument("--template",type=str,help="input template for subtraction")
    parser.add_argument("--cutoff",type=float,help="cutoff counts in the unit of SATCNTS",default=1.0)
    parser.add_argument("--radius",type=int,help="radius to apply cutoff",default=10)
    parser.add_argument("--method",type=str,help="what method...kernels,pca?",default=None)
    parser.add_argument("--kernels",type=str,help="what kernels to convolve?",default=None)
    parser.add_argument("--boundary",type=str,help="boundary to do the image differncing", default=None)
    parser.add_argument("--maskreg",type=str,help="mask a region in the image",default=None)
    parser.add_argument("--sqrt",type=str,help="use sqrt of the ivar", default=False)
    parser.add_argument("--effout",type=str,help="output effective file",default=None)
    parser.add_argument("--plot",type=str,help="show plot or not?", default=False)
    parser.add_argument("--rdnoise",type=str, help="read noise from the image header", default=False)
    args = parser.parse_args()

    do_subtraction(args.image,args.template,cutoff=args.cutoff,radius=args.radius,boundary=args.boundary,
maskreg=args.maskreg,kerneltype=args.kernels,method=args.method,sqrt=args.sqrt,plot=args.plot,effout=args.effout,rdnoise=args.rdnoise)

def do_subtraction(imagefile,templatefile,cutoff=1.0,radius=10,boundary=None,maskreg=None, kerneltype=None,method=None,sqrt=False,plot=False,effout=None,rdnoise=False):


    print "processing", imagefile

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

    image=fits.open(imagefile)
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
    template=fits.open(templatefile)
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
        #- readnoise_i=np.median(Zs)-np.percentile(Zs,15.865)
        #- readnoise_t=np.median(Zt)-np.percentile(Zt,15.865)
        readnoise_i=6.
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
    image_base=str.split(imagefile,'_c.fit')
    image_break=str.split(imagefile,'_')
    temp_base=str.split(templatefile,'_c.fit')  
  
    # mse estimate: 
    mse=1./(np.size(diffimage)) * diffimage.ravel().T.dot(diffimage.ravel())
    print "mse", mse
    #- Results
    finalsc=diffimage##*np.sum(Zs)
    offcenters=finalsc

    # Quality metrics

    ii = np.where(sc_variance>0)
    dof=Zs.size+convtemp.size-convtemp.shape[0]
    print "dof",dof        
    chisq=np.sum(diffimage[ii]**2/sc_variance[ii])/dof
    
    #- fraction of the variance recovered.
    
    R_var=1.0-np.var(diffimage[ii])/np.var(Zs[ii])
    
    print "diff_R2: ",R_var

    diff_imagename=image_base[0]+'-sub_c.fit'
    diff_header=temphdr
    diff_header["MJD"]=imghdr["MJD"]
    diff_header["EXPTIME"]=imghdr["EXPTIME"]
    diff_header["EFFTIME"]=imghdr["EFFTIME"]
    diff_header["DATE-OBS"]=imghdr["DATE-OBS"]
    diff_header["OBSTIME"]=imghdr["OBSTIME"]
    
    #- now for the convolved template(reference)

    refc_image=temp_base[0]+'-'+image_break[0]+'-'+image_break[2]+'-refc_c.fit'
    refchdr=temphdr

    #if R_var > 0.8 : #- not sure if this is okay all the time but keeping as a criteria
    
    fits.writeto(diff_imagename,diffimage,clobber=True,header=diff_header)
    if effout is not None:
        fits.writeto(effout,efftemplate,clobber=True,header=diff_header) #- use same header
    
    print "wrote differenced image", diff_imagename
    
    if plot:
        ax1=plt.subplot(121,projection='3d')
        #ax1.gca(projection='3d')
        X=np.linspace(int(KK.pixelsize)*-1,int(Kern.pixelsize),Kern.pixelsize)
        Y=np.linspace(int(KK.pixelsize)*-1,int(Kern.pixelsize),Kern.pixelsize)
        Xs, Ys = np.meshgrid(X, Y)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        sc_x=np.linspace(0,Zs.shape[0]-1,Zs.shape[0])[Zs.shape[0]/2-40:Zs.shape[0]/2+40]
        sc_y=np.linspace(0,Zs.shape[1]-1,Zs.shape[1])[Zs.shape[0]/2-40:Zs.shape[0]/2+40]

        SCX,SCY=np.meshgrid(sc_x,sc_y)

        surf = ax1.plot_surface(SCX, SCY,finalsc[Zs.shape[0]/2-40:Zs.shape[0]/2+40,Zs.shape[0]/2-40:Zs.shape[0]/2+40],rstride=1, cstride=1,   cmap=cm.Accent, linewidth=0.2)

        ax2=plt.subplot(122)
        
        refpixel=Zs.shape[0]/2
        ax2.step(np.arange(int(Zs.shape[0]/2-40),int(Zs.shape[0]/2+40)),Zs[Zs.shape[0]/2-40:Zs.shape[0]/2+40,refpixel],label='Data')
        ax2.step(np.arange(int(Zs.shape[0]/2-40),int(Zs.shape[0]/2+40)),efftemplate[Zs.shape[0]/2-40:Zs.shape[0]/2+40,refpixel],label='Model Host')
        ax2.step(np.arange(int(Zs.shape[0]/2-40),int(Zs.shape[0]/2+40)),diffimage[Zs.shape[0]/2-40:Zs.shape[0]/2+40,refpixel],label='Residual')

        ax2.text(0.3,0.7, r"$\chi^2/dof = %.2f$"%chisq, verticalalignment='bottom', horizontalalignment='right',transform=ax2.transAxes)
        ax2.set_xlabel("Pixels (relative position)")
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.tick_right()
        ax2.set_ylabel("Counts")
        plt.legend()
        plt.show()

if __name__=='__main__':
    main()
