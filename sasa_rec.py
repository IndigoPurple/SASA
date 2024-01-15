import time
import math
import skimage
import numpy as np
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
# from packages.vnlnet.test import vnlnet
from packages.ffdnet.test_ffdnet_ipol import ffdnet_vdenoiser
from packages.fastdvdnet.test_fastdvdnet import fastdvdnet_denoiser
from utils import (A_, At_, psnr)
if skimage.__version__ < '0.18':
    from skimage.measure import (compare_psnr, compare_ssim)
else: # skimage.measure deprecated in version 0.18 ( -> skimage.metrics )
    import skimage.metrics.peak_signal_noise_ratio as compare_psnr
    import skimage.metrics.structural_similarity   as compare_ssim



def GAP_TV_rec(y,Phi,A, At,Phi_sum, maxiter, step_size, weight, row, col, ColT, X_ori):
    y1 = np.zeros((row,col))
    begin_time = time.time()
    f = At(y,Phi)
    for ni in range(maxiter):
        fb = A(f,Phi)
        y1 = y1+ (y-fb)
        f  = f + np.multiply(step_size, At( np.divide(y1-fb,Phi_sum),Phi ))
        f = denoise_tv_chambolle(f, weight,n_iter_max=30,multichannel=True)
    
        if (ni+1)%5 == 0:
            # mse = np.mean(np.sum((y-A(f,Phi))**2,axis=(0,1)))
            end_time = time.time()
            p = psnr(f, X_ori)
            s = compare_ssim(f,X_ori,multichannel=True)
            # print("GAP-TV: Iteration %3d, PSNR = %2.2f dB,"
            #   " time = %3.1fs."
            #   % (ni+1, p, end_time-begin_time))
    # return f
    return p, s

def ADMM_TV_rec(y,Phi,A, At,Phi_sum, maxiter, step_size, weight, row, col, ColT, eta,X_ori):
    #y1 = np.zeros((row,col))
    begin_time = time.time()
    theta = At(y,Phi)
    v =theta
    b = np.zeros((row,col,ColT))
    for ni in range(maxiter):
        yb = A(theta+b,Phi)
        #y1 = y1+ (y-fb)
        v  = (theta+b) + np.multiply(step_size, At( np.divide(y-yb,Phi_sum+eta),Phi ))
        #vmb = v-b
        theta = denoise_tv_chambolle(v-b, weight,n_iter_max=30,multichannel=True)
        
        b = b-(v-theta)
        weight = 0.999*weight
        eta = 0.998 * eta
        
        if (ni+1)%5 == 0:
            # mse = np.mean(np.sum((y-A(v,Phi))**2,axis=(0,1)))
            end_time = time.time()
            p = psnr(v, X_ori)
            s = compare_ssim(v, X_ori, multichannel=True)
            # print("ADMM-TV: Iteration %3d, PSNR = %2.2f dB,"
            #   " time = %3.1fs."
            #   % (ni+1, p, end_time-begin_time))
    # return v
    return p,s