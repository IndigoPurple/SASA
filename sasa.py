from sasa_utils import *
from sasa_rec import (GAP_TV_rec, ADMM_TV_rec)

det_num = 30

iframe = 0
nframe = 1
# nframe = meas.shape[2]
MAXB = 255.

## [2.5] GAP/ADMM-FFDNet
### [2.5.1] GAP-FFDNet (FFDNet-based frame-wise video denoising)
projmeth = 'gap'  # projection method
_lambda = 1  # regularization factor
accelerate = True  # enable accelerated version of GAP
denoiser = 'ffdnet'  # video non-local network
noise_estimate = False  # disable noise estimation for GAP

iter_max = 40  # maximum number of iterations
iter_max0 = list(range(1, iter_max + 1))
sigma = [60 / 255 * pow(0.971, k - 1) for k in iter_max0]  # pre-set noise standard deviation
# sigma    = [12/255, 6/255] # pre-set noise standard deviation
# iter_max = [10,10] # maximum number of iterations
useGPU = True  # use GPU

data_list = ['aerial32', 'crash32', 'kobe', 'traffic']
for data_id in range(len(data_list)):
    meas_orig, mask, orig_t = read_data(data_list[data_id])

    sasa_meas, sasa_mask = generate_meas_mask(meas_orig, mask, orig_t, det_num)

    psum1,psum2,psum3,psum4 = 0,0,0,0
    ssum1,ssum2,ssum3,ssum4 = 0,0,0,0

    meas_num = sasa_mask.shape[0]
    for mask_i in range(meas_num):
        # print(mask_i)

        # sasa
        mask = sasa_mask[mask_i]
        meas = sasa_meas[:, :, mask_i][:, :, np.newaxis]
        orig = orig_t[:, :, mask_i * 8:8 * (mask_i + 1)]

        # common parameters and pre-calculation for PnP
        # define forward model and its transpose
        # A  = lambda x :  A_(x, mask) # forward model function handle
        # At = lambda y : At_(y, mask) # transpose of forward model

        mask_sum = np.sum(mask, axis=2)
        mask_sum[mask_sum == 0] = 1
        [row, col, ColT] = mask.shape

        # %%
        ## [2.3] GAP/ADMM-TV
        ### [2.3.1] GAP-TV
        projmeth = 'gap'  # projection method
        _lambda = 1  # regularization factor
        accelerate = True  # enable accelerated version of GAP
        denoiser = 'tv'  # total variation (TV)
        iter_max = 40  # maximum number of iterations
        tv_weight = 0.3  # TV denoising weight (larger for smoother but slower)
        tv_iter_max = 5  # TV denoising maximum number of iterations each
        step_size = 1
        eta = 1e-8

        # ADMM_TV_rec(y,Phi,A, At,Phi_sum, maxiter, step_size, weight, row, col, ColT, eta,X_ori)
        # only run the frist measurement
        y = meas[:, :, 0]
        X_ori = orig[:, :, 0:ColT]

        # ADMM-TV
        p_admm_tv, s_admm_tv = ADMM_TV_rec(y / 255, mask, A_, At_, mask_sum, iter_max, step_size, tv_weight, row, col, ColT,
                                           eta, X_ori / 255)

        psum1 += p_admm_tv
        ssum1 += s_admm_tv

        p_gap_tv, s_gap_tv = GAP_TV_rec(y / 255, mask, A_, At_, mask_sum, iter_max, step_size, tv_weight, row, col, ColT,
                                        X_ori / 255)
        psum2 += p_gap_tv
        ssum2 += s_gap_tv

    print('-----data: %s------para: %d----test----' % (data_list[data_id], det_num))
    print(psum1 / meas_num, ssum1 / meas_num)
    print(psum2 / meas_num, ssum2 / meas_num)
