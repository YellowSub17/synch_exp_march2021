

import numpy as np
import scipy.ndimage as sdn
import h5py

import hdf5plugin
import matplotlib.pyplot as plt

from PIL import Image
import os
from skimage.transform import warp_polar

from plot_fns import plot_d_polar_cor

#well f2 dataset 9 frame 102
EIGER_nx = 1062
EIGER_ny = 1028
MAX_PX_COUNT = 4294967295




def polar_plot( data, nr, nth, rmin, rmax, thmin, thmax, cenx, ceny, submean=False ):

    # r and theta arrays
    rarr = np.outer( np.arange(nr)*(rmax-rmin)/float(nr) + rmin, np.ones(nth) )
    tharr = np.outer( np.ones(nr), np.arange(nth)*(thmax-thmin)/float(nth) + thmin)
    newx = rarr*np.cos( tharr ) + cenx
    newy = rarr*np.sin( tharr ) + ceny
    newdata = sdn.map_coordinates( data, [newx.flatten(), newy.flatten()], order=3 )

    out = newdata.reshape( nr, nth )
    if submean == True:
        out = self.polar_plot_subtract_rmean( out  )

    return out



def polar_plot2(data, nr, nth, rmin, rmax, thmin, thmax, cenx, ceny):

    x = warp_polar( data, center=(cenx,ceny), radius=rmax)
    return np.rot90(x, k=3)









def polarplot_angular_correlation(  polar, polar2=None):
    fpolar = np.fft.fft( polar, axis=1 )

    if polar2 != None:
        fpolar2 = np.fft.fft( polar2, axis=1)
        out = np.fft.ifft( fpolar2.conjugate() * fpolar, axis=1 )
    else:
        out = np.fft.ifft( fpolar.conjugate() * fpolar, axis=1 )
    return np.real(out)




def polarplot_angular_intershell_correlation( polar, polar2=None):

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
    assert i<len(h5s), 'i<len(h5s)'
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
    mask_pol = polar_plot2(mask, 500,720,0, 500, 0, 360, int(EIGER_nx/2), int(EIGER_ny/2))

    mask_cor = polarplot_angular_correlation(mask_pol)

#     plt.figure()
    # plt.imshow(mask)

    # plt.figure()
    # plt.imshow(mask_pol)

    # plt.figure()
    # plt.imshow(mask_cor)


    d_sum = read_h5s(i=1)

    d_pol = polar_plot2(d_sum, 500,720,0,500,0,360,  int(EIGER_nx/2), int(EIGER_ny/2))

    d_cor = polarplot_angular_correlation(d_pol)

    # d_q_ave = np.mean(d_cor, axis=1)
    # d_cor -= np.outer(d_q_ave, np.ones(360))



    d_cor = mask_correction(d_cor.astype(mask_cor.dtype), mask_cor)

    plot_d_polar_cor(d_sum, d_pol, d_cor)


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

    # plt.figure()
    # plt.imshow(y, origin='lower', extent=[0,360, 0, 500])
    # plt.title('Unwrapped Test Image (polar_plot)')

    plt.show()

