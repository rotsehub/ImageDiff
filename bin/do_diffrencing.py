#! /usr/bin/env python

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from imagediff.difference import do_subtraction
import numpy as np
import astropy.io.fits as fits
import argparse
import scipy
from scipy.stats import norm

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
    parser.add_argument("--outfile",type=str,help="differenced image name", default=None)
    parser.add_argument("--effout",type=str,help="output effective file",default=None)
    parser.add_argument("--plot",type=str,help="show plot or not?", default=False)
    parser.add_argument("--rdnoise",type=str, help="read noise from the image header", default=False)
    parser.add_argument("--varlimit",type=float,help="variance limit to be expected", default=0.8)

    args = parser.parse_args()


    print "processing", args.image
    image=fits.open(args.image)

    print "Reference ", args.template
    template=fits.open(args.template)

    diffimage, diffvar, efftemplate, Zs, R_var,chisq = do_subtraction(image,template, cutoff=args.cutoff, radius=args.radius,boundary=args.boundary, maskreg=args.maskreg, kerneltype=args.kernels, method=args.method, sqrt=args.sqrt, rdnoise=args.rdnoise)

    image_base=str.split(args.image,'_c.fit')
    image_break=str.split(args.image,'_')
    temp_base=str.split(args.template,'_c.fit') 

    imghdr=image[0].header
    temphdr=template[0].header

    diff_imagename=image_base[0]+'-sub_c.fit'
    diff_header=temphdr
    diff_header["MJD"]=imghdr["MJD"]
    diff_header["EXPTIME"]=imghdr["EXPTIME"]
    diff_header["EFFTIME"]=imghdr["EFFTIME"]
    diff_header["DATE-OBS"]=imghdr["DATE-OBS"]
    diff_header["OBSTIME"]=imghdr["OBSTIME"]

    #write to the file if variance meets expectation;
    print "diff_R2: ",R_var
    
    if R_var > args.varlimit:
        if args.outfile is not None:
            outfilename=args.outfilename
        else: outfilename=diff_imagename

        fits.writeto(outfilename,diffimage,clobber=True,header=diff_header)
        print "wrote differenced image", outfilename

        if args.effout is not None:
            fits.writeto(args.effout,efftemplate,clobber=True,header=diff_header) #- use same header
            print "wrote model image", args.effout
    else:
        print "INFO: Not enough variance in the model.... Not writing to file!"

    if args.plot:
        import matplotlib.mlab as mlab
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax1=plt.subplot(132,projection='3d')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')

        sc_x=np.linspace(0,Zs.shape[0]-1,Zs.shape[0])[Zs.shape[0]/2-40:Zs.shape[0]/2+40]
        sc_y=np.linspace(0,Zs.shape[1]-1,Zs.shape[1])[Zs.shape[0]/2-40:Zs.shape[0]/2+40]

        SCX,SCY=np.meshgrid(sc_x,sc_y)

        surf = ax1.plot_surface(SCX, SCY,diffimage[Zs.shape[0]/2-40:Zs.shape[0]/2+40,Zs.shape[0]/2-40:Zs.shape[0]/2+40],rstride=1, cstride=1,   cmap=cm.Accent, linewidth=0.2)
        #ax1.text(0.7,0.7, r"$R = %.2f$"%R_var, verticalalignment='bottom', horizontalalignment='right',transform=ax1.transAxes,size=2)
        ax1.set_title(r"$R = %.2f$"%R_var,fontsize=20)
        ax2=plt.subplot(131)
        
        refpixel=Zs.shape[0]/2
        ax2.step(np.arange(int(Zs.shape[0]/2-40),int(Zs.shape[0]/2+40)),Zs[Zs.shape[0]/2-40:Zs.shape[0]/2+40,refpixel],label='Data')
        ax2.step(np.arange(int(Zs.shape[0]/2-40),int(Zs.shape[0]/2+40)),efftemplate[Zs.shape[0]/2-40:Zs.shape[0]/2+40,refpixel],label='Model')
        ax2.step(np.arange(int(Zs.shape[0]/2-40),int(Zs.shape[0]/2+40)),diffimage[Zs.shape[0]/2-40:Zs.shape[0]/2+40,refpixel],label='Residual')

        ax2.text(0.85,0.7, r"$\chi^2/dof = %.2f$"%chisq, verticalalignment='bottom', horizontalalignment='right',transform=ax2.transAxes,fontsize=18)
        ax2.set_xlabel("Pixels (relative position)",fontsize=18)
        ax2.set_ylabel("Counts",fontsize=18)
        ylim0=np.min(Zs[Zs.shape[0]/2-40:Zs.shape[0]/2+40,refpixel])-10
        ylim1=np.max(Zs[Zs.shape[0]/2-40:Zs.shape[0]/2+40,refpixel])+10
        ax2.set_ylim(ylim0,ylim1)
        ax2.legend(fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        gd_diff=diffvar>0.
        print "Mean diffimage", np.mean(diffimage)
        devs=(diffimage)/np.sqrt(diffvar)
        #devs=devs[ss]
        print devs.shape
        meandevs=np.mean(devs)
        #sigmadevs=np.median(devs)-np.percentile(devs,15.865)
        sigmadevs=np.std(devs)
        print "Mean devs", meandevs
        print "Sigma devs", sigmadevs
        
        (mu,sig)=norm.fit(devs)
        print "norm fit of devs", mu, sig
        binsz=0.3
        i0, i1 = int( np.min(devs) / binsz) - 1, int( np.max(devs) / binsz) + 1
        rng = tuple( binsz*np.array([i0,i1]) )
        print i0,i1
        nbin = i1-i0
        hist, edges = np.histogram(devs, range=rng, bins=nbin)
        xhist = (edges[1:] + edges[:-1])/2.
        ax3=plt.subplot(133)
        ax3.hist(xhist, color='blue', bins=edges, weights=hist)#, histtype='step')
        # PDF for Gaussian
        area = binsz * np.sum(hist)

        xppf = np.linspace(scipy.stats.norm.ppf(0.0001), scipy.stats.norm.ppf(0.9999), 100)
        xx=np.linspace(-6,6,100)
        gaussfit=mlab.normpdf(xx,meandevs,sigmadevs)
        ax3.plot(xx,area*gaussfit,'r-',alpha=5.0,linewidth=2,label=r'$\mathcal{N}(%.2f,%.2f)$'%(meandevs,sigmadevs))
        ax3.plot(xppf, area*scipy.stats.norm.pdf(xppf), 'k-', alpha=5.0, linewidth=2, label=r'$\mathcal{N}(0,1)$')

        ax3.set_xlabel(r'Residual/$\sigma$',fontsize=18)
        ax3.set_ylabel('No. of pixels',fontsize=18)
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.tick_right()
        ax3.set_xlim(-5,5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.legend(frameon=False,fontsize=18,loc=2)
        #plt.tight_layout()
        plt.show()

if __name__=='__main__':
    main()
