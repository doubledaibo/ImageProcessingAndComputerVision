'''
        ENGG5104 1-2 Richardson-Lucy Deconvolution
        @daibo
        @1155053920
'''
import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm as cm

def rlDeconv(B, PSF):
    threshold = 1e-4
    pad_w = 30
    # Pad border to avoid artifacts
    I = np.pad(B, ((pad_w, pad_w), (pad_w, pad_w), (0,0)),'edge')

    B = I.copy()

    rotate_PSF = cv2.flip(PSF, -1)

    numChannel = I.shape[2]
    newI = I.copy()
    iterIdx = 0
    while True:
        # do RL for each channel
        for j in xrange(numChannel):
    	       convI = cv2.filter2D(I[:,:,j], -1, PSF)
               convI = B[:,:,j] / convI
               convI = cv2.filter2D(convI, -1, rotate_PSF)
               newI[:,:,j] = I[:,:,j] * convI
        diff = np.average(np.fabs(I - newI))
        iterIdx += 1
        print "Iter " + str(iterIdx) + ": Improvement: " + str(diff)
        I = newI.copy()
        # effect of deblur is not significant anymore
        if diff <= threshold:
            break
    print "Done..."
    I = I[pad_w : -pad_w, pad_w : -pad_w]

    return I


if __name__ == '__main__':
    gt = cv2.imread('./misc/lena_gray.bmp').astype('double')
    gt = gt / 255.0

    # You can change to other PSF
    PSF = sio.loadmat('./misc/psf.mat')['PSF']

    # Generate blur image
    B = cv2.filter2D(gt, -1, PSF);

    # Deconvolve image using RL
    I = rlDeconv(B, PSF)

    plt.figure("Richardson-Lucy Deconvolution")
    plt.subplot(131)
    plt.imshow(gt[:, :, [2, 1, 0]])
    plt.title("Original Image")

    plt.subplot(132)
    plt.imshow(B[:, :, [2, 1, 0]])
    plt.title("Blurred Image")

    plt.subplot(133)
    plt.imshow(I[:, :, [2, 1, 0]])
    plt.title("Deblurred Image")
    plt.show()
