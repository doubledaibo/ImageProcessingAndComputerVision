'''
    ENGG5104 1-4 Unsharp Masking
    @daibo
    @1155053920
'''
import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm as cm

def unSharpmask(image, K, A):
    # make sure input is a gray-scale image
    if np.size(image, 2) > 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    sharp_mask = cv2.filter2D(image, -1, K)

    image = image.astype("uint16")
    result = sharp_mask + (A - 1) * image

    idx = result > 255
    result[idx] = 255

    return sharp_mask, result


if __name__ == '__main__':
    im = cv2.imread('./misc/lena_gray.bmp')

    A = 2
    k = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) # generate Laplacian operator

    mask, result = unSharpmask(im, k, A)
    # Show result, OpenCV is BGR-channel, matplotlib is RGB-channel
    # Or:
    #   cv2.imshow('output',result)
    #   cv2.waitKey(0)
    plt.figure("Unsharp Masking")
    plt.subplot(131)
    plt.imshow(im)
    plt.title("Original Image")
    plt.subplot(132)
    plt.imshow(mask, cmap = cm.Greys_r)
    plt.title("Mask")
    plt.subplot(133)
    plt.imshow(result, cmap = cm.Greys_r);
    plt.title("Sharped Image")
    plt.show()
