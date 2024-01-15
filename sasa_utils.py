''' Utilities '''
import math
import numpy as np
import h5py
import scipy.io as sio
import cv2

def A_(x, Phi):
    '''
    Forward model of snapshot compressive imaging (SCI), where multiple coded
    frames are collapsed into a snapshot measurement.
    '''
    return np.sum(x*Phi, axis=2)  # element-wise product

def At_(y, Phi):
    '''
    Tanspose of the forward model.
    '''
    (nrow, ncol, nmask) = Phi.shape
    x = np.zeros((nrow, ncol, nmask))
    for nt in range(nmask):
         x[:,:,nt] = np.multiply(y, Phi[:,:,nt])
    return x
    #return np.multiply(np.repeat(y[:,:,np.newaxis],Phi.shape[2],axis=2), Phi)

def psnr(ref, img):
    '''
    Peak signal-to-noise ratio (PSNR).
    '''
    mse = np.mean( (ref - img) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def generate_meas_mask(meas, mask, orig, max_detections, draw=False, data=None):
    model = 'model/'

    # initialize OpenCV's objectness saliency detector and set the path
    # to the input model files
    saliency = cv2.saliency.ObjectnessBING_create()
    saliency.setTrainingPath(model)

    # load the input meas
    meas_num = meas.shape[2]
    sasa_meas = [meas[:,:,0]]
    sasa_mask = [mask]

    mask_prob = np.zeros((256, 256))
    max_dete = max_detections

    mea = meas[:, :, 0]/8/255
    mea = mea[...,np.newaxis]

    # cv2.imshow('mea', mea)
    # cv2.waitKey(0)

    mea = cv2.cvtColor(mea,cv2.COLOR_GRAY2RGB)

    for meas_i in range(1,meas_num):
        print('measurement id: ',meas_i)

        # compute the bounding box predictions used to indicate saliency
        (success, saliencyMap) = saliency.computeSaliency(mea)
        numDetections = saliencyMap.shape[0]

        # print("numDetections: %d", numDetections)

        # loop over the detections
        for i in range(0, min(numDetections, max_dete)):
            # extract the bounding box coordinates
            (startX, startY, endX, endY) = saliencyMap[i].flatten()

            # randomly generate a color for the object and draw it on the image
            output = mea.copy()*255
            color = np.random.randint(0, 255, size=(3,))
            color = [int(c) for c in color]
            cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)

            if draw:
                cv2.imwrite('./img/maps_mask/%s/maps_%d_%d.png'%(data,meas_i,i), output)
            # show the output image
            # cv2.imshow("Image", output)
            # cv2.waitKey(0)

            # print((startX, startY, endX, endY))
            mask_prob[startY - 1:endY, startX - 1:endX] += 1

        # print(np.max(mask_prob), np.min(mask_prob), np.mean(mask_prob))
        mask_prob /= max_dete
        # cv2.imshow("Mask Prob", mask_prob)
        # cv2.waitKey(0)

        mask_rand = np.random.random((8, 256, 256))
        masks = []
        for i in range(8):
            mask_flag = mask_rand[i] < mask_prob
            mask = np.zeros((256, 256))
            mask[mask_flag == True] = 1
            masks.append(mask)
            # print(mask.shape)
        # print(mask)
        # print(np.sum(mask) / (256 * 256))
        # cv2.imshow("Mask", mask)
        # cv2.waitKey(0)

        # print(masks.shape)
        # cv2.imshow("Masks", np.sum(masks, axis=0) / 8)
        # cv2.waitKey(0)

        current_gt = orig[:,:,8*meas_i:8*(meas_i+1)]
        masks = np.array(masks).transpose(1,2,0)

        mea = A_(current_gt,masks)
        sasa_meas.append(mea)
        sasa_mask.append(masks)

        mea = meas[:, :, 0] / 8 / 255
        mea = mea[..., np.newaxis]

        # cv2.imshow('mea', mea)
        # cv2.waitKey(0)

        mea = cv2.cvtColor(mea, cv2.COLOR_GRAY2RGB)
        if draw:
            # sasa_meas = np.array(sasa_meas).transpose(1, 2, 0)
            for i in range(len(sasa_meas)):
                cv2.imwrite('./img/maps_mask/%s/meas_%d.png' % (data, i), sasa_meas[i]/8)
                for j in range(8):
                    cv2.imwrite('./img/maps_mask/%s/mask_%d_%d.png' % (data, i,j), sasa_mask[i][:,:,j] *255)

    return np.array(sasa_meas).transpose(1,2,0), np.array(sasa_mask)

def read_data(data_name):
    # [0] environment configuration
    datasetdir = './dataset/'  # dataset

    # datname = 'starfish_c16_48'    # name of the dataset

    matfile = datasetdir + data_name + '_cacti.mat'  # path of the .mat data file

    # %%
    from scipy.io.matlab.mio import _open_file
    from scipy.io.matlab.miobase import get_matfile_version

    # [1] load data
    if get_matfile_version(_open_file(matfile, appendmat=True)[0])[0] < 2:  # MATLAB .mat v7.2 or lower versions
        file = sio.loadmat(matfile)  # for '-v7.2' and lower version of .mat file (MATLAB)
        order = 'K'  # [order] keep as the default order in Python/numpy
        meas = np.float32(file['meas'])
        mask = np.float32(file['mask'])
        orig = np.float32(file['orig'])
    else:  # MATLAB .mat v7.3
        file = h5py.File(matfile, 'r')  # for '-v7.3' .mat file (MATLAB)
        order = 'F'  # [order] switch to MATLAB array order
        meas = np.float32(file['meas'], order=order).transpose()
        mask = np.float32(file['mask'], order=order).transpose()
        orig = np.float32(file['orig'], order=order).transpose()

    return meas, mask, orig