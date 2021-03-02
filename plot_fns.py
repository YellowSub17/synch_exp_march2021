import matplotlib.pyplot as plt
import numpy as np

def plot_d_polar_cor(d_sum, d_pol, d_cor):
    plt.figure()
    plt.imshow(d_sum)
    plt.title('Sum data')

    plt.figure()
    plt.imshow(d_pol, origin='lower', extent=[0,360, 0, 500])
    plt.title('data polar')
    plt.xlabel('theta')
    plt.ylabel('r [pix]')

    plt.figure()
    plt.plot(np.sum(np.abs(d_pol),axis=1))
    plt.title('sum data polar')
    plt.xlabel('r [pix]')
    plt.ylabel('intensity')

    plt.figure()
    plt.imshow(np.abs(d_cor),origin='lower', extent=[0,360, 0, 500])
    plt.title('abs data correlation q1=q2')
    plt.xlabel('theta')
    plt.ylabel('r [pix]')

    plt.figure()
    plt.plot(np.sum(np.abs(d_cor),axis=1))
    plt.title('sum abs data correlation')
    plt.xlabel('r [pix]')
    plt.ylabel('intensity')


    plt.figure()
    plt.plot(np.abs(d_pol[86,:]))
    plt.title('data polar r=86')
    plt.xlabel('r [pix]')
    plt.ylabel('intensity')


    plt.figure()
    plt.plot(np.abs(d_cor[86,:]))
    plt.title('data correlation r=86')
    plt.xlabel('theta')
    plt.ylabel('intensity')

    plt.figure()
    plt.plot(np.abs(d_pol[151,:]))
    plt.title('data polar r=151')
    plt.xlabel('theta')
    plt.ylabel('intensity')


    plt.figure()
    plt.plot(np.abs(d_cor[151,:]))
    plt.title('data correlation r=151')
    plt.xlabel('theta')
    plt.ylabel('intensity')



