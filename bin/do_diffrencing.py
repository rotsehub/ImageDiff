#! /usr/bin/env python

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import gauss_hermite_psf as gh
import astropy.io.fits as fits
import get_kernels as gk
import get_delta_psf as gdp
import get_regress_matrix_coeff as gr
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
    args = parser.parse_args()

    do_subtraction(args.image,args.template,cutoff=args.cutoff,radius=args.radius,boundary=args.boundary,maskreg=args.maskreg,kernels=args.kernels,method=args.method)

def do_subtraction(imagefile,templatefile,cutoff=1.0,radius=10,boundary=None,maskreg=None, kernels=None,method=None):
    print "processing", imagefile
    X=np.linspace(-4,4,9)
    Y=np.linspace(-4,4,9)

    #- Define kernels
    if kernels is None: #- Gaussian poly as default
        kernels = "gauss_poly"
        allk=gk.total_kernels(X,Y,sigma=[1.,1.5,3.5],orders=[4,3,2])
    if kernels == "gauss_poly":
        allk=gk.total_kernels(X,Y,sigma=[1.,1.5,3.5],orders=[4,3,2])
    if kernels == "gauss_hermite":
        #allk=gh.new_kernels(X,Y,sigmax=1.5,sigmay=1.5,m=4)
        allk=gh.new_kernels(X,Y,m=7)
    if kernels=="delta":
        allk=gdp.get_delta_kernels(X,Y)


    if kernels not in ["gauss_poly","gauss_hermite","delta"]:
        raise ValueError("kernels not matching or implemented, define a matching kernel")
    print "using kernel", kernels
#For delta_function psf
    R2=gdp.make_R_matrix(X.shape[0])
    H=R2.T.dot(R2)
    

    image=fits.open(imagefile)
    mask=np.ones_like(image[0].data)
    print image[0].data.shape
    print "mask shape", mask.shape
    if maskreg is not None:
        print "Mask reg.", maskreg
        mxlo,mxhi,mylo,myhi=map(int, maskreg.split(','))
        mask[mxlo:mxhi,mylo:myhi]=0
        #print image[0].data[mxlo:mxhi,mylo:myhi]

    if boundary is not None:
        xlo,xhi,ylo,yhi=map(int, boundary.split(','))
    else:
        xlo,xhi,ylo,yhi=0,image[0].data.shape[0],0,image[0].data.shape[1]
   
    image_data=image[0].data[xlo:xhi,ylo:yhi]#*mask[xlo:xhi,ylo:yhi]    
    sky_image=gr.get_background_level(image_data)
    Zs=image_data-sky_image #- are you subtracting signal also?
    std_dev_img=np.std(Zs)
    imghdr=image[0].header    
    if imghdr["PC001001"] < 0:  # Coordinate rotation matrix opposite!
        Zs=np.rot90(np.rot90(Zs))
        print "rotating"  

    #per5_i=np.percentile(Zs,5)
    #per95_i=np.percentile(Zs,95)
    #maskrms_i=np.std(Zs[(Zs<per95_i)])# & (Zs<per95_i)]) #- 90% confidence
    #if maskreg is not None:
    #    Zs[mxlo:mxhi,mylo:myhi]+=np.random.normal(0,scale=maskrms_i/2,size=(mxhi-mxlo,myhi-mylo))
    mean_img=np.mean(Zs)

    variance_in_data=np.var(Zs) # overall variance in the data

    
   
    template=fits.open(templatefile)
    template_data=template[0].data[xlo:xhi,ylo:yhi]#*mask[xlo:xhi,ylo:yhi]
    
    sky_template=gr.get_background_level(template_data)
    Zt=template_data-sky_template
    std_dev_temp=np.std(Zt)
    #per5_t=np.percentile(Zt,5)
    #per95_t=np.percentile(Zt,95)
    #maskrms_t=np.std(Zt[(Zt<per95_t)])# & (Zt<per95_t)])
    #if maskreg is not None:
    #    Zt[mxlo:mxhi,mylo:myhi]+=np.random.normal(0,scale=maskrms_t/2,size=(mxhi-mxlo,myhi-mylo))

    temphdr=template[0].header
    print "STD", std_dev_img,std_dev_temp
    #print maskrms_i,maskrms_t


    #- Get needed header parameters to propagate
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
    print k_picked.shape
    print "No. of pixels above cutoff", k_t[0].shape[0]
    #- Add some noise to the masked regions:
    #print "count", Zt[Zt==1.0e-30]
    rmsi=np.std(Zs[(Zs>np.percentile(Zs,5)) & (Zs< np.percentile(Zs,95))])
    rmst=np.std(Zt[(Zt>np.percentile(Zt,5)) & (Zt < np.percentile(Zt,95))])
    Zt[Zt==1.0e-30]=np.random.normal(0,rmst,size=k_picked.shape)
    Zs[Zs==1.0e-30]=np.random.normal(0,rmsi,size=k_picked.shape)
    #Zt[Zt==1.0e-30]+=np.random.normal(0,np.std(Zt),size=Zt[Zt==1.0e-30].shape[0])

    #print Zt[Zt==1.0e-30]
    readnoise_t=temphdr["BSTDDEV"]
    readnoise_i=imghdr["BSTDDEV"]

    
    exptime_t=temphdr["EXPTIME"]
    exptime_i=imghdr["EXPTIME"]
    
    #sc_variance=Zs.clip(0.0)+readnoise_i**2
    #temp_variance=Zt.clip(0.0)+readnoise_t**2
    
    sc_variance=image_data.clip(0.0)+readnoise_i**2
    temp_variance=template_data.clip(0.0)+readnoise_t**2

    sc=Zs#.clip(0.0)

    temp=Zt#.clip(0)
    
    #- Denoise template first use bilateral
    from skimage.restoration import denoise_bilateral
    #- requirement:
    #- grayscale 2d float should be -1 to +1
    #- can't be negative

    #tmp_temp=Zt.clip(0.0)
    #sum_temp=tmp_temp.sum()
    #temp=tmp_temp/sum_temp
    #temp=denoise_bilateral(temp,multichannel=False)
    #temp*=sum_temp        

    convtemp=gr.get_convolve_kernel(temp,allk) #2D [nvec,nobs]
    print "convtemp-shape", convtemp.shape
    lamb=1.#- regularization parameter
    H=lamb*H*std_dev_img
    #coeff1,lsq1=gr.solve_coeff(convtemp,variance,sc)
    subreg=int(sc.shape[0]/300)+1 #- dividing into subregions
    print "subreg",subreg
    #coeffs=np.zeros(subreg,subreg,allk.shape[2])
    efftemplate=np.zeros_like(sc)
    n=300


    if method == "PCA":
 
        #- using pca:
        #modes=gr.get_PCA_basis(convtemp)

        #- use empca
        print "using pca"
        import empca
        nx,ny=sc.shape[0:2]
        n_im=convtemp.shape[0]
        data=np.array([convtemp[i,:] for i in range(convtemp.shape[0])],'f')
        #- may be add background
        #data+=sky_template.ravel()
        var_data=data.clip(0.)+readnoise_t**2
        #wt=np.ones_like(data)
        wt=1./var_data
        #- Add image to the data list
        #sc+=sky_image
        sc_var=sc.clip(0.)+readnoise_i**2
        all_data=np.vstack((data,sc.ravel()))
        all_wt=np.vstack((wt,1./sc_var.ravel()))
        model=empca.empca(all_data,all_wt**0.5,niter=15,nvec=10,smooth=0) # will output only eigen vectors default(=5)
        print model.eigvec.shape
        effdata=model.eigvec.T.dot(model.coeff.T)
        efftemplate=model.eigvec.T.dot(model.coeff[-1]).reshape(nx,ny)
        dof = all_data.size-10*nx-10*(n_im+1)
        print "dof",dof
        Chisq=np.sum((effdata-all_data.T)**2*all_wt.T)/dof
        print "Chisq_pca", Chisq
    else: #- kernel based
        #for ii in range(subreg): #-x
        #    for jj in range(subreg): #-y
                #print n*ii,n*jj
        #        xsol,iCov,y=gr.solve_for_coeff(convtemp[:,n*n*ii:n*n*(ii+1)-1],sc[n*ii:n*(ii+1)-1,n*jj:n*(jj+1)-1].ravel(),sc_variance[n*ii:n*(ii+1)-1,n*jj:n*(jj+1)-1].ravel()) #- Use H if regualarization needed
        #        efftemplate[n*ii:n*(ii+1)-1,n*jj:n*(jj+1)-1]=convtemp[:,n*n*ii:n*n*(ii+1)-1].T.dot(xsol).reshape(n,n)

        #- First set matrix, vectors etc...

        A=convtemp.T
        ivar=1./sc_variance.ravel()#**0.5
        from scipy.sparse import spdiags
        wt=spdiags(ivar,0,ivar.size,ivar.size) #- only image weights
        b=sc.ravel()
        print A.shape, b.shape,wt.shape
        xsol=gr.solve_for_coeff(A,b,W=wt)
        efftemplate=convtemp.T.dot(xsol).reshape(sc.shape)
        
    #- Integral of kernels:
    #kernel_comb=allk.dot(xsol)

    diffimage=(sc-efftemplate)*mask[xlo:xhi,ylo:yhi] #+sky_image #- add back the sky background
    if maskreg is not None: 
        diffrms=np.std(diffimage[(diffimage>np.percentile(diffimage,5)) & (diffimage<np.percentile(diffimage,95))])
        print "diff rms",diffrms, 'rmsi',rmsi
        diffimage[mxlo:mxhi,mylo:myhi]=np.random.normal(scale=diffrms,size=(mxhi-mxlo,myhi-mylo))
    image_base=str.split(imagefile,'_c.fit')
    image_break=str.split(imagefile,'_')
    temp_base=str.split(templatefile,'_c.fit')  
  
    # rmse estimate:
    rmse=np.sqrt(np.sum((sc[0:100,0:100]-efftemplate[0:100,0:100])**2/sc[0:100,0:100].size))  

    #- Results and Plot

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Xs, Ys = np.meshgrid(X, Y)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    finalsc=diffimage##*np.sum(Zs)
    offcenters=finalsc
    diff_bkg=gr.get_background_level(finalsc)
    diff_bkg_rms=np.std(finalsc)

    #- fraction of the variance recovered.
    
    R_var=1.0-np.var(diffimage)/variance_in_data
    
    print "diff_R2: ",R_var

    print "rmse: ", rmse
    
    #print "Chisq", np.sum(diffimage**2/variance_in_data)/dof
    #print "Mean Difference:", np.mean(offcenters),'+/-',np.std(offcenters) # demand N(0,1) like distribution
    #ax.set_zlim(-500,2000)

    #- write difference image file if mean and rms are resonably good.
    #- get the headers correct first
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
    #fits.writeto(refc_image,efftemplate,clobber=True,header=temphdr) # writing image data instead of template convolved as in idl case
    #else: 
    #    print "Differencing does not seem okay. Not writing to file for ", image_base[0]

    #- write kernels if needed (useful for old image_differencing)
    #fits.writeto('gauss_hermite_19x19_kernels.fits',allk,clobber=True)

    sc_x=np.linspace(0,sc.shape[0]-1,sc.shape[0])[sc.shape[0]/2-40:sc.shape[0]/2+40]
    sc_y=np.linspace(0,sc.shape[1]-1,sc.shape[1])[sc.shape[0]/2-40:sc.shape[0]/2+40]
    print sc.shape
    SCX,SCY=np.meshgrid(sc_x,sc_y)

    surf = ax.plot_surface(SCX, SCY,finalsc[sc.shape[0]/2-40:sc.shape[0]/2+40,sc.shape[0]/2-40:sc.shape[0]/2+40],rstride=1, cstride=1,   cmap=cm.Accent, linewidth=0.2)
    #plt.show()
    #surf=ax.plot_surface(Xs,Ys,kernel_comb,rstride=1,cstride=1,cmap=cm.coolwarm)
    #plt.show()
    #plt.imshow(kernel_comb,'gray',interpolation=None)
    plt.show()

if __name__=='__main__':
    main()
