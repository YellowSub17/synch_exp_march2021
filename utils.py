

import numpy as np
import scipy.ndimage as sdn
import h5py

import hdf5plugin
import matplotlib.pyplot as plt

from PIL import Image
import os
from skimage.transform import warp_polar

import plot_fns

#well f2 dataset 9 frame 102
EIGER_nx = 1062
EIGER_ny = 1028
MAX_PX_COUNT = 4294967295




def to_polar(data, nr, nth, rmin, rmax, thmin, thmax, cenx, ceny):

    x = warp_polar( data, center=(cenx,ceny), radius=rmax)
    return np.rot90(x, k=3)





def polar_angular_correlation(  polar, polar2=None):
    fpolar = np.fft.fft( polar, axis=1 )

    if polar2 != None:
        fpolar2 = np.fft.fft( polar2, axis=1)
        out = np.fft.ifft( fpolar2.conjugate() * fpolar, axis=1 )
    else:
        out = np.fft.ifft( fpolar.conjugate() * fpolar, axis=1 )
    return np.real(out)




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

# def mask_correction(  corr, maskcorr ):
    # corr *= 1.0/maskcorr
    # return corr


def read_h5s(i=1):


    h5s = os.listdir('data')
    assert i<=len(h5s), 'i>len(h5s)'
    mask = make_mask()
    mask_loc = np.where(mask==0)
    sum_data = np.zeros( (EIGER_nx, EIGER_ny))

    for h5 in range(i):

        f = h5py.File(f'data/{h5s[h5]}')

        d = np.array(f['entry/data/data'])[::20,:,:]


        sum_data += np.sum(d, 0)

    return sum_data*mask


def make_mask():

    h5s = os.listdir('data')
    f = h5py.File(f'data/{h5s[0]}')

    d = np.array(f['entry/data/data'])
    mask = np.ones((EIGER_nx, EIGER_ny))
    max_loc = np.where(d[0,...] == MAX_PX_COUNT)
    mask[max_loc] =0

    return mask









if __name__ =='__main__':



    mask = make_mask()
    mask_pol = to_polar(mask, 500,720,0, 500, 0, 360, int(EIGER_nx/2), int(EIGER_ny/2))
    mask_cor = polar_angular_correlation(mask_pol)


    #####Plot Mask
    # plot_fns.plot_im(mask, title='Mask')
    # plot_fns.plot_polar(mask_pol, title='Mask Polar')
    # plot_fns.plot_polar(mask_cor, title='Mask Corr.')


    d_sum = read_h5s(i=1)
    d_pol = to_polar(d_sum, 500,720,0,500,0,360,  int(EIGER_nx/2), int(EIGER_ny/2))
    d_cor = polar_angular_correlation(d_pol)
    d_cor = mask_correction(d_cor.astype(mask_cor.dtype), mask_cor)

    #####Plot Data
    plot_fns.plot_im(d_sum, 'Data')

    #####Plot Data (Polar)
    plot_fns.plot_polar(d_pol, 'Data Polar')
    plot_fns.plot_q(d_pol, 115,'Data Polar, q=115')
    plot_fns.plot_sumtheta(d_pol, 'Data Polar, sumTheta')

    #####Plot Data (Corr)
    plot_fns.plot_polar(d_cor, 'Data Corr.')
    plot_fns.plot_q(d_cor, 115,'Data Corr., q=115')
    plot_fns.plot_sumtheta(d_cor, 'Data Corr., sumTheta')





#     im = Image.open('testing2.png')
    # mask = np.asarray(im)[:,:,0]
    # x = polar_plot2(mask, 100,180,0, 500, 0, 360, int(1062/2), int(1028/2))
    # y = polar_plot(mask, 100,180,0, 500, 0, 360, int(1062/2), int(1028/2))
    # plt.figure()
    # plt.imshow(mask)
    # plt.title('Test Image')
    # plt.figure()
    # plt.imshow(x, origin='lower', extent=[0,360, 0, 500])
    # plt.title('Unwrapped Test Image (polar_plot2)')



    plt.show()

