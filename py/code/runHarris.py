'''
    ENGG5104 1-3 Harris Corner Detector
    @daibo
    @1155053920
'''
import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt

def harrisdetector(image, k, t):
    if np.size(image, 2) > 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h = np.size(image, 0)
    w = np.size(image, 1)
    # padding
    image = np.lib.pad(image, ((k, k), (k, k)), "edge")

    # derivatives
    fx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    fy = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    Ix = cv2.filter2D(image, -1, fx)
    Iy = cv2.filter2D(image, -1, fy)
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    IxIy = Ix * Iy

    # find corners
    ptr_x = []
    ptr_y = []

    for i in xrange(k, h + k):
        for j in xrange(k, w + k):
            # find A
            A11 = 0
            A12 = 0
            A22 = 0
            for u in xrange(i - k, i + k + 1):
                for v in xrange(j - k, j + k + 1):
                    A11 += Ix2[u, v]
                    A12 += IxIy[u, v]
                    A22 += Iy2[u, v]
            A = np.array([[A11, A12], [A12, A22]])
            retval, eigenvalues, eigenvectors = cv2.eigen(A)
            if eigenvalues[0] > t and eigenvalues[1] > t:
                ptr_x.append(j - k)
                ptr_y.append(i - k)

    result = [ptr_x, ptr_y]
    return result

if __name__ == '__main__':
    k = 0     # change to your value
    t = 1e-3  # change to your value

    I = cv2.imread('./misc/corner_gray.png').astype("float32")
    print "k: " + str(k) + " t : " + str(t)
    fr = harrisdetector(I, k, t)

    plt.figure("Harris Corner Detector")
    plt.imshow(I[:,:,[2, 1, 0]])
    # plot harris points overlaid on input image
    plt.scatter(x=fr[0], y=fr[1], c='r', s=40)

    # show
    plt.show()
