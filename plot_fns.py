import matplotlib.pyplot as plt
import numpy as np



def plot_sumtheta(pol, q=np.array([]), title='', new_fig=True):
    if new_fig:
        plt.figure()
    if q.size == 0:
        plt.plot(np.sum(np.abs(pol),axis=1))
        plt.xlabel('r [pix]')
    else:
        plt.plot(q,np.sum(np.abs(pol),axis=1))
        plt.xlabel('q [A-1]')

    plt.title(f'{title}')
    plt.ylabel('intensity')

def plot_q(pol, iq, title='',new_fig=True):
    if new_fig:
        plt.figure()
    plt.plot(np.abs(pol[iq,:]))
    plt.title(f'{title}')
    plt.xlabel('theta')
    plt.ylabel('intensity')

def plot_polar(pol, q=None, title='',new_fig=True, tmax=360):
    if new_fig:
        plt.figure()
    if q is None:
        plt.imshow(pol, origin='lower', extent=[0,tmax, 0, pol.shape[0]],aspect='auto')
        plt.ylabel('r [pix]')
    else:
        plt.imshow(pol, origin='lower', extent=[0,tmax, 0, q[-1]],aspect='auto')
        plt.ylabel('q [A-1]')

    plt.xlabel('theta')
    plt.title(f'{title}')

def plot_im(im, title='',new_fig=True):
    if new_fig:
        plt.figure()
    plt.imshow(im)
    plt.title(f'{title}')







# def plot_d_polar_cor(d_sum, d_pol, d_cor):
    # plt.figure()
    # plt.imshow(d_sum)
    # plt.title('Sum data')

    # plt.figure()
    # plt.imshow(d_pol, origin='lower', extent=[0,360, 0, 500])
    # plt.title('data polar')
    # plt.xlabel('theta')
    # plt.ylabel('r [pix]')

    # # plt.figure()
    # # plt.plot(np.sum(np.abs(d_pol),axis=1))
    # # plt.title('sum data polar')
    # # plt.xlabel('r [pix]')
    # # plt.ylabel('intensity')
    # plot_sum_thru_theta(d_pol, title='sum data polar')
    # plot_sum_thru_theta(d_cor, title='sum data correlation')

    # plt.figure()
    # plt.imshow(np.abs(d_cor),origin='lower', extent=[0,360, 0, 500])
    # plt.title('abs data correlation q1=q2')
    # plt.xlabel('theta')
    # plt.ylabel('r [pix]')

# #     plt.figure()
    # # plt.plot(np.sum(np.abs(d_cor),axis=1))
    # # plt.title('sum abs data correlation')
    # # plt.xlabel('r [pix]')
    # # plt.ylabel('intensity')


    # plt.figure()
    # plt.plot(np.abs(d_pol[86,:]))
    # plt.title('data polar r=86')
    # plt.xlabel('r [pix]')
    # plt.ylabel('intensity')


    # plt.figure()
    # plt.plot(np.abs(d_cor[86,:]))
    # plt.title('data correlation r=86')
    # plt.xlabel('theta')
    # plt.ylabel('intensity')

    # plt.figure()
    # plt.plot(np.abs(d_pol[115,:]))
    # plt.title('data polar r=115')
    # plt.xlabel('theta')
    # plt.ylabel('intensity')


    # plt.figure()
    # plt.plot(np.abs(d_cor[115,:]))
    # plt.title('data correlation r=115')
    # plt.xlabel('theta')
    # plt.ylabel('intensity')



