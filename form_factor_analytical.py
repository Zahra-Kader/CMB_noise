#calculates dc/drho and K(theta) and saves data  files to be used in Hankel transform to find form factor (in form_factor_hankel.py)
#actually ended up calculating form factor here - rewrite fourier space integral as convolution and find product in real space
#finish changing 'tt convergence' to be more general

# import sys ; sys.argv=['form_factor_analytical.py','600','10'] ; execfile('form_factor_analytical.py') 

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
import scal_power_spectra
import inputs
import os

args=sys.argv
nside=int(args[1])      #Jet had 512. Heather changed to make ACT-like 2048 nside and 20 fov_deg
fov_deg=float(args[2])

expt='ref' # 'advact'  #planck, ref_foregrounds, ref
spec='tt'
est='conv'

if (expt[0:3]=='ref'):
    working_dir=inputs.working_dir_base + 'ref' + '/fov'+str(int(fov_deg))
    if not os.path.exists(working_dir):
		os.makedirs(working_dir)
else:
    working_dir=inputs.working_dir_base+expt+'/fov'+str(int(fov_deg))
    if not os.path.exists(working_dir):
		os.makedirs(working_dir)
        
working_dir_FormFactor=inputs.working_dir_base+expt+'/fov'+str(int(fov_deg))+'/FormFactor'
if not os.path.exists(working_dir_FormFactor):
	os.makedirs(working_dir_FormFactor)
    
#working_dir='/Users/Heather/Documents/HeatherVersionPlots/FormFactor/IntegralByConvolution/Datafiles'

noisy_maps=False#True remind yourself what this does!!
if expt=='advact':
    theta_fwhm=1.4 #arcmin, 7for planck

if expt=='ref':
    theta_fwhm=1.5 #arcmin, 7for planck
# This is extracted from getNoiseTT in scal_power_spectra - using 'ref' expt for now - GENERALISE THIS TO BE READ IN
    
#if expt =='pix':
#    sigma_p=1.0 # uK arcmin
#    theta_fwhm_pix = 1.0 # arcmin
        
lmax= 2048 # 8192 # 8192*4 ##4096#8192#*2
delta_l= 1 # 2*4 ##4#1#2 #should be 1 for 4096, I am just trying to get pol code to run
n=lmax/delta_l*2
l_vec,cl_tt, cl_ee, cl_te, cl_tt_lens, cl_ee_lens, cl_bb_lens, cl_te_lens, cl_kk=scal_power_spectra.spectra()
if expt=='ref_foregrounds':
    n_l_tt=(scal_power_spectra.getNoiseTT(l_vec, 'ref'))
else:
    n_l_tt_beam=(scal_power_spectra.getNoiseTT(l_vec, expt))#'planck')
    n_l_tt_pixel = (scal_power_spectra.getNoiseTT(l_vec, 'pix')) # set sigma_p = 1 so additional noise factor, just pixel window function
    n_l_tt=n_l_tt_beam #* n_l_tt_pixel # QUICK FIX - only choose this option if including pixel window function for real maps
    
n_l_pol=n_l_tt*2.

# deltal = 2    Max= 5.0304402836593426e-08
# deltal = 1    Max= 5.03069643639e-08
#

if expt=='ref_foregrounds':
    cib_poisson=l_vec**2*1e-6
    n_l_tt+=cib_poisson/(l_vec*(l_vec+1)/(2*np.pi))




def diffSpectraManual(ls, cs):
    cspl=spline1d(ls, cs)
    dcdl=np.zeros(ls.shape[0])
    for i in range (ls.shape[0]):
        dcdl[i]=(cspl(ls[i]*1.01)-cspl(ls[i]*0.99))/(ls[i]*0.02)
    return dcdl

#returns K(l)=g(l)/N for a given power spectrum and estimator
def getFilter(spec, est):
    if spec=='tt':
        a=(cl_tt/(cl_tt_lens+n_l_tt)**2)
        dcdl=diffSpectraManual(l_vec, cl_tt) #should it be cl_tt_lens??
        dlncdlnl=dcdl*l_vec/cl_tt             #cl_tt_lens??
        #dlncdlnl[0]=0
        if est=='conv':
            d=dlncdlnl+2.
            g=a*d
            N_integrand=g*cl_tt*d*l_vec/(2*np.pi)
        elif est=='shear_plus' or est=='shear_cross':
            print 'tt shear plus'
            d=dlncdlnl
            g=a*d
            N_integrand=0.5*g*cl_tt*d*l_vec/(2*np.pi)
        N=np.trapz(N_integrand)
        filt_vec=g/N
        filt_vec[l_vec.size-1]=0.
        filt_vec[0]=filt_vec[1]

    elif spec=='ee':
        a=(cl_ee/(cl_ee_lens+n_l_pol)**2)
        dcdl=diffSpectraManual(l_vec, cl_ee) #should it be cl_ee_lens??
        dlncdlnl=dcdl*l_vec/cl_ee             #cl_ee_lens??
        if est=='conv':
            d=dlncdlnl+2.
            g=a*d
            N_integrand=g*cl_ee*d*l_vec/(2*np.pi)
        elif est=='shear_plus' or est=='shear_cross':
            d=dlncdlnl
            g=a*d
            N_integrand=0.5*g*cl_ee*d*l_vec/(2*np.pi)
        N=np.trapz(N_integrand)
        filt_vec=g/N
        filt_vec[l_vec.size-1]=0.
        filt_vec[l_vec.size-2]=0.


    elif spec=='te':
        a=1/((cl_te_lens)**2+(cl_ee_lens+n_l_pol)*(cl_tt_lens+n_l_tt))
        dcdl=diffSpectraManual(l_vec, cl_te) #should it be cl_ee_lens??
        dcdlnl=dcdl*l_vec             #cl_ee_lens??
        if est=='conv':
            d=dcdlnl+2.*cl_te
            g=a*d
            N_integrand=g*d*l_vec/(2*np.pi)
        elif est=='shear_plus' or est=='shear_cross':
            d=dcdlnl
            g=a*d
            N_integrand=0.5*g*d*l_vec/(2*np.pi)
        N=np.trapz(N_integrand)
        filt_vec=g/N
        filt_vec[l_vec.size-1]=0.
        filt_vec[l_vec.size-2]=0.

    elif spec=='eb':
        if est=='conv':
            filt_vec=np.zeros(l_vec.shape)
            N_integrand=np.ones(l_vec.shape)
        elif est=='shear_plus' or est=='shear_cross':
            filt_vec=cl_ee/((cl_ee_lens+n_l_pol)*(cl_bb_lens+n_l_pol))
            N_integrand=0.5*filt_vec*cl_ee*l_vec/(2*np.pi)
        N=np.trapz(N_integrand)
        filt_vec/=N

    elif spec=='tb':
        if est=='conv':
            filt_vec=np.zeros(l_vec.shape)
            N_integrand=np.ones(l_vec.shape)
            print 'tb conv filt', filt_vec
        elif est=='shear_plus' or est=='shear_cross':
            filt_vec=cl_te/((cl_tt_lens+n_l_tt)*(cl_bb_lens+n_l_pol))
            N_integrand=0.5*filt_vec*cl_te*l_vec/(2*np.pi)
        N=np.trapz(N_integrand)
        filt_vec/=N
    return filt_vec


def getSpecSpline(spec):
    if spec=='tt':
        cl=cl_tt
    elif spec=='ee':
        cl=cl_ee
    elif spec=='te':
        cl=cl_te
    elif spec=='eb' or spec=='tb':
        cl=np.zeros(l_vec.shape)
    return(spline1d(l_vec, cl))
""" Nans coz of inf
def getNoiseSpline(spec):
    if spec=='tt':
        nl=n_l_tt
    elif spec=='ee':
        nl=n_l_tt
    else:
        print 'I don\'t know what noise power spectrum should be for cross spectra...'
        import sys
        sys.exit(1)
    return(spline1d(l_vec, nl))
"""
    
    
K=getFilter(spec, est)
lK=np.zeros((K.shape[0],3))
lK[:,0]=l_vec
lK[:,2]=K
#np.savetxt(spec+'_'+est+'_'+expt+'_datafile.txt', lK)

"""
w=np.loadtxt('wmap_tt_spec.txt')
l_w, c_w=w[:,0], w[:,1]
cw_spl=spline1d(l_w, c_w*2*np.pi/(l_w*(l_w+1)))
"""

if spec=='tt' or spec=='ee' or spec=='te':
    clspl=getSpecSpline(spec)
    #nlspl=getNoiseSpline(spec)
elif spec=='tb':
    clspl=getSpecSpline('te')
elif spec=='eb':
    clspl=getSpecSpline('ee')
    #nlspl=getNoiseSpline('ee') ?????

Kspl=spline1d(np.concatenate((np.array([0,1]),l_vec)), np.concatenate((np.array([K[0]]), np.array([K[0]]),K)))


lx=np.arange(-lmax,lmax, delta_l)
ly=np.arange(-lmax,lmax, delta_l)

print lx

lxs, lys=np.meshgrid(lx, ly)

l=np.sqrt(lxs**2+lys**2)

clgrid=np.zeros(l.shape)
#nlgrid=np.zeros(l.shape)
Kgrid=np.zeros(l.shape)
#c_w_grid=np.zeros(r.shape)
cos_2theta_grid=(lxs**2-lys**2)/(lxs.astype(float)**2+lys**2)
sin_2theta_grid=(2*lxs*lys)/(lxs.astype(float)**2+lys**2)

cos_2theta_grid[n/2,n/2]=1
sin_2theta_grid[n/2,n/2]=1

sigma_b=theta_fwhm*(1/60.)*(np.pi/180)/(np.sqrt(8*np.log(2)))    #np.log is natural logarithm
if(noisy_maps):
    beam_grid=np.exp(-0.5*(l*sigma_b)**2)
else:
    beam_grid=np.ones(l.shape)

for i,l1 in enumerate(l[0,:]):
    clgrid[:,i]=clspl(l[:,i])
    #nlgrid[:,i]=nlspl(l[:,i])
    #c_w_grid[:,i]=w_spl(l[:,i])
    Kgrid[:,i]=Kspl(l[:,i])

sigma_pix=1.5 # 5.5 #(microK per arcmin), detector noise in pixel of side theta_fwhm?
# This is extracted from getNoiseTT in scal_power_spectra - using 'ref' expt for now - GENERALISE THIS TO BE READ IN

#c_plus_n_grid=clgrid+nlgrid
extra_term_in_noisy_form_fac=np.trapz(np.trapz(Kgrid*(beam_grid**2*clgrid+np.ones(l.shape)*(theta_fwhm*sigma_pix))))
print 'extra_term_in_noisy_form_fac:', extra_term_in_noisy_form_fac

term_to_subtract=np.trapz(np.trapz(Kgrid*clgrid))    #est applied to unlensed maps
print 'term to subtract:', term_to_subtract


#if noisy_maps is true, this includes the beam smearing effect. If not, this just multiplies by 1
clgrid*=beam_grid
Kgrid*=beam_grid

if (spec=='tt' or spec=='ee' or spec=='te') and est=='shear_plus':
    Kgrid=cos_2theta_grid*Kgrid
elif (spec=='tt' or spec=='ee' or spec=='te') and est=='shear_cross':
    Kgrid=sin_2theta_grid*Kgrid
elif (spec=='tb' or spec=='eb')and est=='shear_plus':
    Kgrid=sin_2theta_grid*Kgrid
elif (spec=='tb' or spec=='eb')and est=='shear_cross':
    Kgrid=-cos_2theta_grid*Kgrid


#check eb and tb - I got this from Jet's code but not sure where that was from!
if spec=='tt':
    clgrid=np.fft.fftshift(clgrid)
    Kgrid=np.fft.fftshift(Kgrid)

if spec=='ee' or spec=='te' or spec=='tb' or spec=='eb':
    clgrid_cos=np.fft.fftshift(clgrid*cos_2theta_grid)
    clgrid_sin=np.fft.fftshift(clgrid*sin_2theta_grid)
    Kgrid_cos=np.fft.fftshift(Kgrid*cos_2theta_grid)
    Kgrid_sin=np.fft.fftshift(Kgrid*sin_2theta_grid)
    print 'C and K initialised for te/ee'
    
    c_theta_cos=np.fft.ifft2(clgrid_cos)
    c_theta_sin=np.fft.ifft2(clgrid_sin)
    K_theta_cos=np.fft.ifft2(Kgrid_cos)
    K_theta_sin=np.fft.ifft2(Kgrid_sin)
    print 'real space C and K done te/ee'
    """
    plt.imshow(np.real(np.fft.fftshift(c_theta_cos)))
    plt.colorbar()
    plt.title('ifft of cos2phi C')
    plt.show()
    """
    del clgrid_cos, clgrid_sin, Kgrid_cos, Kgrid_sin





#c_w_grid=np.fft.fftshift(c_w_grid)
"""
plt.plot(np.fft.fftshift(l)[:n/2, 0], clgrid[:n/2,0])
plt.title('c(l)')
plt.show()"""
#plt.imshow(c_w_grid[0:2,0:2])
#plt.colorbar()
#plt.show()

"""
plt.imshow(clgrid[0:2,8190:])
plt.colorbar()
plt.show()


plt.imshow(clgrid[8190:,0:2])
plt.colorbar()
plt.show()

plt.imshow(clgrid[8190:,8190:])
plt.colorbar()
plt.show()
"""

theta=360*np.arange(0, n/2)/float(n*delta_l)
dx=dy=delta_theta=(theta[1]-theta[0])*np.pi/180
if spec=='tt':
    Ctheta=np.fft.ifft2(clgrid)#np.real()
    Ktheta=np.fft.ifft2(Kgrid)#np.real()
    #c_w_theta=np.real(np.fft.ifft2(c_w_grid))
    print Ctheta
    
    c=Ctheta[0:n/2, 0]
    Ks=Ktheta[0:n/2, 0]
    """
    plt.plot(theta,c)
    plt.title('c in real space')
    plt.show()


    plt.plot(theta,Ks)
    plt.title('K in real space')
    plt.show()
    

    w_corr=np.loadtxt('wmap_tt_corr.txt')
    plt.plot(w_corr[:,0], w_corr[:,1])
    plt.plot(theta,(c_w_theta[0:lmax, 0]))
    plt.show()


    tc=np.zeros((c.shape[0],2))
    tc[:,0]=theta
    tc[:,1]=c
    np.savetxt(working_dir+'/c_tt_theta_lmax'+str(lmax), tc)

    dx=dy=theta[1]-theta[0]*np.pi/180       #radians
    dc=np.gradient(c, (dx,))

    tdc=tc
    tdc[:,0]=theta
    tdc[:,1]=dc

    plt.plot(theta, dc)
    plt.title('dc(r)/dr')
    plt.xlabel('theta in degrees')
    plt.show()

    np.savetxt(working_dir+'/diff_c_tt_theta_lmax'+str(lmax), tdc)

    tK=np.zeros((c.shape[0],2))
    tK[:,0]=theta
    tK[:,1]=Ks
    np.savetxt(working_dir+'/K_tt_theta_lmax'+str(lmax), tK)
    dK=np.gradient(Ks, (dx,))

    tdK=tK
    tdK[:,1]=dK

    np.savetxt(working_dir+'/diff_K_tt_theta_lmax'+str(lmax), tdK)
    """







"""
tx=ty=360*np.arange(-n/2, n/2)/float(n*delta_l)
txs, tys=np.meshgrid(tx,ty)

rho=np.sqrt(txs**2+tys**2)

Kdc=Ks*dc
d_Kdc=np.gradient(Kdc)
dkdcspl=spline1d(theta, d_Kdc)
"""

"""
plt.plot(theta, d_Kdc)
plt.show()"""
"""
dkdc_grid=np.zeros(rho.shape)
for i,r1 in enumerate(rho[0,:]):
    dkdc_grid[:,i]=dkdcspl(rho[:,i])
    """
"""
plt.imshow(dkdc_grid, extent=[-45,45,-45,45])
plt.colorbar()
plt.show()"""

"""
dkdc_grid=np.fft.fftshift(dkdc_grid)
fft_dkdc_grid=np.fft.fft2(dkdc_grid)

Lsquared_F=np.absolute(np.fft.fftshift(fft_dkdc_grid))
plt.imshow(Lsquared_F)
plt.colorbar()
plt.title('L squared times form factor')
plt.show()

FF=Lsquared_F/(l**2)
plt.imshow(FF)
plt.colorbar()
plt.title('form factor')
plt.show()

plt.plot(ly[n/2:], FF[0:n/2,0]/FF[1,0], label='yslice')
plt.xlim(0,1000)
#plt.legend()
plt.title('Form factor for '+spec+' '+est+' estimator for '+expt)
plt.show()
"""



if spec=='tt':
    gradCtheta=np.fft.fftshift(np.gradient(np.fft.fftshift(Ctheta), dx)) #(dx,dy))) , for some reason grad won't work when given (dx,dy) but I'm pretty sure it uses dx for dy
    dcdth_times_K_grid_x=gradCtheta[0]*Ktheta
    dcdth_times_K_grid_y=gradCtheta[1]*Ktheta
 
    #not enough resolution to get correct derivative at 0 - put it in manually
    dcdth_times_K_grid_x[0,0]=0
    dcdth_times_K_grid_y[0,0]=0
    
    f_dcdth_K_x=np.fft.fft2(dcdth_times_K_grid_x)
    f_dcdth_K_y=np.fft.fft2(dcdth_times_K_grid_y)
    print 'grad c:',gradCtheta
    print 'der times kx', dcdth_times_K_grid_x
    print 'fft der times kx', f_dcdth_K_x
    Lsquared_times_F=lxs*np.fft.fftshift((f_dcdth_K_x))+lys*np.fft.fftshift((f_dcdth_K_y))


if spec=='ee' or spec=='te':
    dcdth_times_K_grid_x_cos=np.fft.fftshift(np.gradient(np.fft.fftshift(c_theta_cos), (dx,)))[0]*K_theta_cos
    dcdth_times_K_grid_y_cos=np.fft.fftshift(np.gradient(np.fft.fftshift(c_theta_cos), (dx,)))[1]*K_theta_cos
    dcdth_times_K_grid_x_sin=np.fft.fftshift(np.gradient(np.fft.fftshift(c_theta_sin), (dx,)))[0]*K_theta_sin
    dcdth_times_K_grid_y_sin=np.fft.fftshift(np.gradient(np.fft.fftshift(c_theta_sin), (dx,)))[1]*K_theta_sin

    dcdth_times_K_grid_x_cos[0,0]=0
    dcdth_times_K_grid_y_cos[0,0]=0
    dcdth_times_K_grid_x_sin[0,0]=0
    dcdth_times_K_grid_y_sin[0,0]=0
    """
    plt.imshow(np.real(np.fft.fftshift(dcdth_times_K_grid_x_cos)))
    plt.colorbar()
    plt.title('grad(ifft(cos2phi C))*ifft(cos2phi K) x comp')
    plt.show()
    """
    print 'dcdtheta k found in RS'
    
    f_dcdth_K_x_cos=np.fft.fft2(dcdth_times_K_grid_x_cos)
    f_dcdth_K_y_cos=np.fft.fft2(dcdth_times_K_grid_y_cos)
    f_dcdth_K_x_sin=np.fft.fft2(dcdth_times_K_grid_x_sin)
    f_dcdth_K_y_sin=np.fft.fft2(dcdth_times_K_grid_y_sin)
    
    print 'dcdtheta k transformed to FS'
    """
    plt.imshow(np.real(np.fft.fftshift(f_dcdth_K_x_cos)))
    plt.colorbar()
    plt.title('fft(grad(ifft(cos2phi C))*ifft(cos2phi K)) x comp')
    plt.show()
    """
    del dcdth_times_K_grid_x_cos, dcdth_times_K_grid_y_cos, dcdth_times_K_grid_x_sin, dcdth_times_K_grid_y_sin
    
    Lsquared_times_F=lxs*np.fft.fftshift((f_dcdth_K_x_cos))+lys*np.fft.fftshift((f_dcdth_K_y_cos))+lxs*np.fft.fftshift((f_dcdth_K_x_sin))+lys*np.fft.fftshift((f_dcdth_K_y_sin))
    del f_dcdth_K_x_cos, f_dcdth_K_x_sin, f_dcdth_K_y_cos, f_dcdth_K_y_sin
    print 'found L^2f'

if spec=='eb' or spec=='tb':
    dcdth_times_K_grid_x_1=np.fft.fftshift(np.gradient(np.fft.fftshift(c_theta_sin), (dx,)))[0]*K_theta_cos #sin cos
    dcdth_times_K_grid_y_1=np.fft.fftshift(np.gradient(np.fft.fftshift(c_theta_sin), (dx,)))[1]*K_theta_cos #sin cos
    dcdth_times_K_grid_x_2=np.fft.fftshift(np.gradient(np.fft.fftshift(c_theta_cos), (dx,)))[0]*K_theta_sin   #cos sin
    dcdth_times_K_grid_y_2=np.fft.fftshift(np.gradient(np.fft.fftshift(c_theta_cos), (dx,)))[1]*K_theta_sin   #cos sin

    dcdth_times_K_grid_x_1[0,0]=0
    dcdth_times_K_grid_y_1[0,0]=0
    dcdth_times_K_grid_x_2[0,0]=0
    dcdth_times_K_grid_y_2[0,0]=0

    print 'dcdtheta k found in RS'
    
    f_dcdth_K_x_1=np.fft.fft2(dcdth_times_K_grid_x_1)
    f_dcdth_K_y_1=np.fft.fft2(dcdth_times_K_grid_y_1)
    f_dcdth_K_x_2=np.fft.fft2(dcdth_times_K_grid_x_2)
    f_dcdth_K_y_2=np.fft.fft2(dcdth_times_K_grid_y_2)
    
    print 'dcdtheta k transformed to FS'

    del dcdth_times_K_grid_x_1, dcdth_times_K_grid_y_1, dcdth_times_K_grid_x_2, dcdth_times_K_grid_y_2
    
    Lsquared_times_F=lxs*np.fft.fftshift((f_dcdth_K_x_1))+lys*np.fft.fftshift((f_dcdth_K_y_1))-lxs*np.fft.fftshift((f_dcdth_K_x_2))-lys*np.fft.fftshift((f_dcdth_K_y_2))
    #del f_dcdth_K_x_1, f_dcdth_K_x_1, f_dcdth_K_y_1, f_dcdth_K_y_1
    print 'found L^2f'



"""
plt.plot(theta,dcdth_times_K_grid_x[0:n/2, 0])
plt.plot(theta,dcdth_times_K_grid_y[0,0:n/2])
plt.plot(theta, Ks*dc)
plt.title('K * dc(r)dr')
plt.show()

to_fft=np.gradient(dcdth_times_K_grid_x)[1]+np.gradient(dcdth_times_K_grid_y)[0]

Lsquared_times_F_2=np.absolute(np.fft.fftshift(np.fft.fft2(np.gradient(dcdth_times_K_grid_x)[1])))+np.absolute(np.fft.fftshift(np.fft.fft2(np.gradient(dcdth_times_K_grid_y)[0])))#np.fft.fft2(to_fft)
F_2=np.absolute(Lsquared_times_F_2)/(l**2)
F_shift_2=np.fft.fftshift(F_2)
plt.imshow(F_2, extent=[-lmax, lmax,-lmax,lmax])#,vmin=-max, vmax=max)
plt.colorbar()
plt.show()

plt.plot(ly[n/2:], F_shift_2[0:n/2,0]/F_shift_2[1,0])
plt.title('Form factor for '+spec+' '+est+' estimator for '+expt)
plt.xlabel('L')
plt.show()
"""







F=np.imag(Lsquared_times_F/(l**2)) 
F_shift=np.fft.fftshift(F)


norm=0.5*delta_theta**2

plt.imshow(F/norm, extent=[-lmax, lmax,-lmax,lmax])
plt.colorbar()
plt.title('Form factor for '+spec+' '+est)
plt.show()

F_shift=np.fft.fftshift(F)





"""
    
    #plt.plot(ly[n/2:], np.absolute(F_shift[0:n/2,0]/F_shift[1,0]), label='absolute')
    plt.plot(ly[n/2:], F_shift[0:n/2,0]/F_shift[1,0], label='x slice')
    plt.plot(ly[n/2:], F_shift[0,0:n/2]/F_shift[0,1], label='y slice')
    plt.legend()
    plt.title('Form factor for '+spec+' '+est+' estimator for '+expt)
    plt.xlabel('L')
    plt.show()
    
    #plt.plot(ly[n/2:], np.absolute(F_shift[0:n/2,0]/F_shift[1,0]), label='absolute')
    #plt.plot(ly[n/2:], np.real(F_shift[0:n/2,0]/F_shift[1,0]), label='real')
    plt.plot(ly[n/2:], F_shift[0:n/2,0]/F_shift[1,0], label='x slice')
    plt.plot(ly[n/2:], F_shift[0,0:n/2]/F_shift[0,1], label='y slice')
    plt.legend()
    plt.title('Form factor for '+spec+' '+est+' estimator for '+expt)
    plt.xlabel('L')
    plt.xlim(0,1000)
    plt.show()
    """
"""plt.plot(np.fft.fftshift(Lsquared_times_F)[0:n/2,0], label='Dot product with L in fourier sp')#/np.fft.fftshift(Lsquared_times_F)[1,0])
#plt.plot(np.fft.fftshift(Lsquared_times_F_2)[0:n/2,0]*3e3, label='Grad in realsp (rescaled)')#/np.fft.fftshift(Lsquared_times_F_2)[1,0])
plt.title('L squared times Form factor for '+spec+' '+est+' estimator for '+expt)
plt.legend()
plt.show()"""


ff=np.zeros((n/2))
for i in range(0, n/2):
        ff[i]=F_shift[i,i]


plt.plot(ly[n/2:]*np.sqrt(2),ff/norm, label='45 degree slice')
plt.plot(ly[n/2:], F_shift[0,0:n/2]/norm, label='xslice')
plt.plot(ly[n/2:], F_shift[0:n/2,0]/norm, label='yslice')
plt.xlim(0,1000)
plt.legend()
plt.title('Form factor for '+spec+' '+est+' estimator for '+expt)
plt.show()



plt.plot(ly[n/2:]*np.sqrt(2),ff/norm, label='45 degree slice')
plt.plot(ly[n/2:], F_shift[0,0:n/2]/norm, label='xslice')
plt.plot(ly[n/2:], F_shift[0:n/2,0]/norm, label='yslice')
plt.legend()
plt.title('Form factor for '+spec+' '+est+' estimator for '+expt)
plt.show()

lf=np.zeros((n/2, 2))
lf[:,0]=lx[n/2:]*np.sqrt(2)
lf[:,1]=ff
# lf[:,1]=lf[:,1]/lf[1,1]
if noisy_maps:
    np.savetxt(working_dir+'/ff_'+spec+'_'+est+'_alternative_analytical_'+expt+'_lmax'+str(lmax)+'_deltal'+str(delta_l)+'_other_axis_with_noise', lf)
else:
    np.savetxt(working_dir+'/ff_'+spec+'_'+est+'_alternative_analytical_'+expt+'_lmax'+str(lmax)+'_deltal'+str(delta_l)+'_other_axis', lf)

lf=np.zeros((n/2, 2))
lf[:,0]=lx[n/2:]
#if est=='shear_plus' or est=='conv':
lf[:,1]=F_shift[0,0:n/2]
lf[:,1]=lf[:,1] /lf[1,1] 
print 'deltal =', delta_l, 'Max=', lf[1,1]
#elif est=='shear_cross':
#lf[:,1]=np.real(F_shift[0, 0:n/2]/F_shift[0,1])
np.savetxt(working_dir_FormFactor+'/form_factor_'+spec+'_'+est+expt+'_lmax'+str(lmax)+'_deltal'+str(delta_l)+'_with_noise_'+str(sigma_pix)+'_thetafwhm'+str((theta_fwhm))+'.txt', lf[1:])
if noisy_maps:
    np.savetxt(working_dir+'/ff_'+spec+'_'+est+'_alternative_analytical_'+expt+'_lmax'+str(lmax)+'_deltal'+str(delta_l)+'_with_noise_'+str(sigma_pix)+'_thetafwhm_'+str((theta_fwhm)), lf)
else:
    np.savetxt(working_dir+'/ff_'+spec+'_'+est+'_alternative_analytical_'+expt+'_lmax'+str(lmax)+'_deltal'+str(delta_l), lf)

#lf[:,1]=F_shift[0,0:n/2]
#np.savetxt(working_dir+'/ff_'+spec+'_'+est+'_alternative_analytical_'+expt+'_lmax'+str(lmax)+'_deltal'+str(delta_l)+'_other_axis', lf)

#

plt.plot(lf[:,0],lf[:,1]) 
plt.show()

raise KeyboardInterrupt

if est=='shear_plus':
    plt.imshow(F/(cos_2theta_grid*norm))#, vmax=1e-7, vmin=-1e-7)
    plt.colorbar()
    plt.show()
    #np.savetxt(working_dir+'/ff_full_2D_'+spec+'_'+est+'_alternative_analytical_'+expt+'_lmax'+str(lmax)+'_deltal'+str(delta_l), F/cos_2theta_grid)
#Fp=np.loadtxt(working_dir+'/ff_full_2D_'+spec+'_shear_plus_alternative_analytical_'+expt+'_lmax'+str(lmax)+'_deltal'+str(delta_l))

if est=='shear_cross':
    plt.imshow(F/(sin_2theta_grid*norm))#, vmax=1e-7, vmin=-1e-7)
    plt.colorbar()
    plt.show()
    #np.savetxt(working_dir+'/ff_full_2D_'+spec+'_'+est+'_alternative_analytical_'+expt+'_lmax'+str(lmax)+'_deltal'+str(delta_l), F/sin_2theta_grid)





