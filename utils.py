

import numpy as np
import scipy.ndimage as sdn
import h5py

#import hdf5plugin
import matplotlib.pyplot as plt

from PIL import Image
import os
#from skimage.transform import warp_polar

import plot_fns

#well f2 dataset 9 frame 102
EIGER_nx = 1062
EIGER_ny = 1028
MAX_PX_COUNT = 4294967295




#def to_polar(data, nr, nth, rmin, rmax, thmin, thmax, cenx, ceny):
#
#    x = warp_polar( data, center=(cenx,ceny), radius=rmax)
#    return np.rot90(x, k=3)

def to_polar( data, nr, nth, rmin, rmax, thmin, thmax, cenx, ceny):

        # r and theta arrays
        rarr = np.outer( np.arange(nr)*(rmax-rmin)/float(nr) + rmin, np.ones(nth) )
        tharr = np.outer( np.ones(nr), np.arange(nth)*(thmax-thmin)/float(nth) + thmin)
        newx = rarr*np.cos( tharr ) + cenx
        newy = rarr*np.sin( tharr ) + ceny
        # plt.imshow(rarr)
        # plt.figure()
        # plt.imshow(tharr)
        # plt.figure()

        newdata = sdn.map_coordinates( data, [newx.flatten(), newy.flatten()], order=3 )

        out = newdata.reshape( nr, nth )


        # if sub_tmean:
            # ave = np.mean(out, axis=1)
            # print(ave.shape)
            # sub_a = np.outer(out,np.ones(out.shape[1]))
            # out -= sub_a
        return out



def polar_angular_correlation(  polar, polar2=None, sub_tmean=False):
    fpolar = np.fft.fft( polar, axis=1 )

    if polar2 != None:
        fpolar2 = np.fft.fft( polar2, axis=1)
        out = np.fft.ifft( fpolar2.conjugate() * fpolar, axis=1 )
    else:
        out = np.fft.ifft( fpolar.conjugate() * fpolar, axis=1 )

    out = np.real(out)

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


def sum_h5s(path, i=None):
    '''
    path: path to directory of .h5 files
    i: how many h5 files to read in that directory
    '''

    # print(path)
    h5s = os.listdir(path)
    if i is None:
        i=len(h5s)
    elif i>len(h5s):
        print('number of h5s to sum is greater that number of h5s in path')
        print('summing all h5s in path')
        i=len(h5s)

    sum_data = np.zeros( (EIGER_nx, EIGER_ny))
    end_h5_str= '0'*(5-len(str(len(h5s)-1))) + str(len(h5s)-1)
    print(end_h5_str)

    for h5 in range(i):
        if 'master' in h5s[h5]:
            continue
        if end_h5_str in h5s[h5]:
            continue
        print(f'reading: {h5s[h5]}')

        with h5py.File(f'{path}/{h5s[h5]}') as f:

            d = np.array(f['entry/data/data'])
        sum_data += np.sum(d, 0)

    return sum_data


def make_mask(path):

    h5s = os.listdir(path)
    
    with h5py.File(f'{path}/{h5s[0]}') as f:

        d = np.array(f['entry/data/data'])

    mask = np.ones((EIGER_nx, EIGER_ny))
    max_loc = np.where(d[0,...] == MAX_PX_COUNT)
    mask[max_loc] =0

    return mask

def qscale(npix,pe=18500,z=0.6):
    
    hc = 12398.4  #eV / A
    wl = hc/pe    # wavelength in angstrom
    
    # print("wavelength = ", wl, "A")
    pw = 75e-6   #pixel width
    
    q = np.arange(npix)
    
    qout = 2*np.pi*(2/wl)*np.sin(np.arctan(pw*q/z)/2.0)
    #d = 2*np.pi / qmax
    return qout






if __name__ =='__main__':



    rmin = 80
    q = qscale( 500-rmin ) 

    group='cthot'
    run_start = 69

    mask = make_mask('/data/xfm/data/2021r1/Binns_16777/raw/eiger/ctt/66249_60/')
    mask_pol = to_polar(mask,  500-rmin,720,rmin,500,0,2*np.pi,  int(EIGER_nx/2), int(EIGER_ny/2))
    mask_cor = polar_angular_correlation(mask_pol)

    for run in range(run_start,run_start+1):

        path = f'/data/xfm/data/2021r1/Binns_16777/raw/eiger/{group}/{66189+run}_{run}/'


        #####Read H5
        d_sum = sum_h5s(path)*mask

        
        #####Plot Data
        plot_fns.plot_im(d_sum, f'Data, group {group}, run {run}')


        #####Create polar data
        d_pol = to_polar(d_sum, 500-rmin,720,rmin,500,0,2*np.pi,  int(EIGER_nx/2), int(EIGER_ny/2))

        #####Plot Data (Polar)
        plot_fns.plot_polar(d_pol, title=f'Data Polar, group {group}, run: {run}')
        plot_fns.plot_sumtheta(d_pol, title=f'Data Polar, sumTheta, group {group}, run: {run}')



        #####Create Correlation
        d_cor = polar_angular_correlation(d_pol, sub_tmean=True)
        d_cor = mask_correction(d_cor.astype(mask_cor.dtype), mask_cor)
        #####Plot Correlation
        plot_fns.plot_polar(d_cor, title='Data Corr.' )
        plot_fns.plot_sumtheta(d_cor,  title='Data Corr., sumTheta')
    








#     im = Image.open('testing2.png')
    # mask = np.asarray(im)[:,:,0]
    # x = to_polar2(mask, 100,180,0, 500, 0, 360, int(1062/2), int(1028/2))
    # y = to_polar(mask, 100,180,0, 500, 0, 360, int(1062/2), int(1028/2))
    # plt.figure()
    # plt.imshow(mask)
    # plt.title('Test Image')
    # plt.figure()
    # plt.imshow(x, origin='lower', extent=[0,360, 0, 500])
    # plt.title('Unwrapped Test Image (to_polar2)')



    plt.show()

