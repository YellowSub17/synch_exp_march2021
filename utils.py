

import numpy as np
import scipy.ndimage as sdn
import h5py

import matplotlib.pyplot as plt

from PIL import Image
import os

import plot_fns

EIGER_nx = 1062
EIGER_ny = 1028
MAX_PX_COUNT = 2**32-1





def to_polar( data, nr, nth, rmin, rmax, thmin, thmax, cenx, ceny):
        '''
        data: image to unwrap
        nr: number of theta bins
        nth: number of theta bins
        rmin: minimum r value
        rmax: maximum r value
        thmin: minimum theta value
        thmax: maximum theta value
        cenx: center x pixel of the data
        ceny: center y pixel of the data
        '''

        # r and theta arrays
        rarr = np.outer( np.arange(nr)*(rmax-rmin)/float(nr) + rmin, np.ones(nth) )
        tharr = np.outer( np.ones(nr), np.arange(nth)*(thmax-thmin)/float(nth) + thmin)
        newx = rarr*np.cos( tharr ) + cenx
        newy = rarr*np.sin( tharr ) + ceny

        newdata = sdn.map_coordinates( data, [newx.flatten(), newy.flatten()], order=3 )

        out = newdata.reshape( nr, nth )


        return out



def polar_angular_correlation(  polar, polar2=None, sub_tmean=False):
    '''
    polar: unwraped polar image
    polar2: optional cross correlation polar image
    sub_tmean: optional subtraction of angular average per q
    '''
    fpolar = np.fft.fft( polar, axis=1 )

    if polar2 != None:
        fpolar2 = np.fft.fft( polar2, axis=1)
        out = np.fft.ifft( fpolar2.conjugate() * fpolar, axis=1 )
    else:
        out = np.fft.ifft( fpolar.conjugate() * fpolar, axis=1 )

    
    out = np.real(out)
    
    out = out[:, :int(out.shape[1]/2)]


    if sub_tmean:
        ave = np.mean(out, axis=1)
        sub_a = np.outer(ave,np.ones(out.shape[1]))
        out -= sub_a
    return out





def polar_angular_intershell_correlation( polar, polar2=None):

    fpolar = np.fft.fft( polar, axis=1 )

    if polar2 != None:
        fpolar2 = np.fft.fft( polar2, axis=1)
    else:
        fpolar2 = fpolar

    out = np.zeros( (polar.shape[0],polar.shape[0],polar.shape[1]) )
    for i in np.arange(polar.shape[0]):
        for j in np.arange(polar.shape[0]):
            out[i,j,:] = fpolar[i,:]*fpolar2[j,:].conjugate()
    out = np.fft.ifft( out, axis=2 )

    return out


def mask_correction(  corr, maskcorr ):
    imask = np.where( maskcorr != 0 )
    corr[imask] *= 1.0/maskcorr[imask]
    return corr


def sum_run(path, i=None):

    '''
    path: path to directory of .h5 files
    i: how many h5 files to read in that directory
    '''

    h5s = os.listdir(path)
    assert len(h5s) > 2, 'path must have more than one (+plus master) h5 file'
    h5s.sort()
    h5s = h5s[:-2] #remove master and last h5 (contains date dump)
    print(h5s)

    sum_data = np.zeros( (EIGER_nx, EIGER_ny))

    for h5 in h5s[:i]:
        print(f'reading: {h5}')

        with h5py.File(f'{path}/{h5}') as f:
            d = np.array(f['entry/data/data'])

        sum_data += np.sum(d, 0)

    return sum_data





def correlate_run(path,nr,nth,rmin,rmax,thmin,thmax,cenx,ceny, i=None):

    '''
    path: path to directory of .h5 files
    i: how many h5 files to read in that directory
    '''

    h5s = os.listdir(path)
    assert len(h5s) > 2, 'path must have more than one (+plus master) h5 file'
    h5s.sort()
    h5s = h5s[:-2] #remove master and last h5 (contains date dump)
    print(h5s)

    cor_data = np.zeros( (nr, int(nth/2)))

    for h5 in h5s[:i]:
        print(f'reading: {h5}')

        with h5py.File(f'{path}/{h5}') as f:
            d = np.array(f['entry/data/data'])[:,:,::10]

        for i in range(d.shape[-1]):
            d_pol = to_polar(d[...,i],nr,nth,rmin,rmax,thmin,thmax,cenx,ceny)
            d_cor = polar_angular_correlation(  d_pol, sub_tmean=True)
            cor_data +=d_cor

    return cor_data

def single_frame(path, h5, i=[0]):

    h5s = os.listdir(path)

    with h5py.File(f'{path}/{h5s[h5]}') as f:

        d = np.array(f['entry/data/data'])

    out = np.sum(d[(np.array(i)),...], 0)



    return out

def make_mask(path):

    h5s = os.listdir(path)
    with h5py.File(f'{path}/{h5s[0]}') as f:

        d = np.array(f['entry/data/data'])

    mask = np.ones((EIGER_nx, EIGER_ny))
    max_loc = np.where(d[0,...] == MAX_PX_COUNT)
    mask[max_loc] =0

    return mask

def qscale(npix,pmin=0,pe=18500,z=0.15):
    hc = 12398.4  #eV / A
    wl = hc/pe    # wavelength in angstrom
    # print("wavelength = ", wl, "A")
    pw = 75e-6   #pixel width
    q = np.arange(npix-pmin)+pmin
    qout = 2*np.pi*(2/wl)*np.sin(np.arctan(pw*q/z)/2.0)
    #d = 2*np.pi / qmax
    return qout






if __name__ =='__main__':


    cenx = int(EIGER_nx/2)+0.25
    ceny = int(EIGER_ny/2)

    rmin = 80
    rmax = 300
    q = qscale( rmax,rmin )
    q2d = np.outer( q**4,np.ones(360))

    #groups=['vortmo','vortmo', 'vortmo', 'vortmo', 'vortmo']
    #runs = [76, 77, 78, 79, 80]

    # groups = ['vortmo']
    # runs = [93]

    # groups = ['capillary']
    # runs = [44]

    # groups = ['dlc', 'dlc','dlc', 'dlc']
    # runs = [102, 104, 106,108]

    groups = ['vortmo', 'vortmo']
    runs=[93, 76]

    # groups = ['dlc', '.']
    # runs = [114, 116]

    mask = make_mask('/data/xfm/data/2021r1/Binns_16777/raw/eiger/ctt/66249_60/')
    mask_pol = to_polar(mask,  rmax-rmin,720,rmin,rmax,0,2*np.pi,  int(EIGER_nx/2), int(EIGER_ny/2))
    mask_cor = polar_angular_correlation(mask_pol)


    for run, group in zip(runs,groups):

        path = f'/data/xfm/data/2021r1/Binns_16777/raw/eiger/{group}/{66189+run}_{run}/'

        d_cor = correlate_run(path,rmax-rmin,720,rmin,rmax,0,2*np.pi,cenx,ceny, i=1)
        # #####Read run
        d_sum = sum_run(path, i=1)*mask

        # #####Create polar data
        d_pol = to_polar(d_sum, rmax-rmin,720,rmin,rmax,0,2*np.pi, cenx, ceny)

        plot_fns.plot_polar(d_pol, title=f'Data polar, {run}' , tmax=180, q=q)


        # plot_fns.plot_sumtheta(d_pol, q=q, title=f'{group} {run}', new_fig=False)



        # # #####Create Correlation
        # d_cor = polar_angular_correlation(d_pol, sub_tmean=True)
        d_cor = mask_correction(d_cor.astype(mask_cor.dtype), mask_cor)
        d_cor *= q2d
        # # cth = np.outer(np.ones(len(q)), np.cos(2*np.pi*np.arange(360)/360.0) )
        # # ic = np.where(np.abs(cth)>5e-2)
        # # d_cor[ic] *= 1/np.abs(cth[ic])
        # # d_cor[ic] *= 1.0/np.abs(cth[ic])
        # ####Plot Correlation
        plot_fns.plot_polar(d_cor, title=f'Data Corr, {run}' , tmax=180, q=q)
        # plt.colorbar()
        # plt.clim([np.min(d_cor)*0.1,np.max(d_cor)*0.05])
        # plt.clim([0,np.max(d_cor)*0.05])
        # # plot_fns.plot_sumtheta(d_cor,q=q,  title='Data Corr., sumTheta')
        # # plt.figure()
        # # plt.plot(d_cor[71,:]) 






# #     im = Image.open('testing2.png')
    # # mask = np.asarray(im)[:,:,0]
    # # x = to_polar2(mask, 100,180,0, 500, 0, 360, int(1062/2), int(1028/2))
    # # y = to_polar(mask, 100,180,0, 500, 0, 360, int(1062/2), int(1028/2))
    # # plt.figure()
    # # plt.imshow(mask)
    # # plt.title('Test Image')
    # # plt.figure()
    # # plt.imshow(x, origin='lower', extent=[0,360, 0, 500])
    # # plt.title('Unwrapped Test Image (to_polar2)')



    plt.show()

