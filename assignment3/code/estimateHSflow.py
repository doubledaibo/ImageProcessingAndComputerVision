import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import scipy.sparse.linalg as sslg
import flowTools as ft

def add_nonzero_entry(row, column, data, pR, pC, pV):
    row.append(pR)
    column.append(pC)
    data.append(pV)

def estimateHSflowlayer(frame1, frame2, uv, lam=80, maxwarping=10):
    H, W = frame1.shape
    npixels = H * W

    x, y = np.meshgrid(range(W), range(H))


    # TODO#3: build differential matrix and Laplacian matrix according to
    # image size
    print "Computing Dx, Dy, L..."
    e = np.ones(npixels)
    Dy = sp.spdiags([-e, e], [0, 1], npixels, npixels)
    Dx = sp.spdiags([-e, e], [0, H], npixels, npixels)
    L = np.dot(Dx.transpose(), Dx) + np.dot(Dy.transpose(), Dy)
    # diagonalize
    #L = np.reshape(L, npixels)
    # Kernel to get gradient
    h = np.array([[1, -8, 0, 8, -1]], dtype=np.float32) / 12
    remap = np.zeros([H, W, 2])

    for i in range(maxwarping):

        # TODO#2: warp image using the flow vector
        # an example is in runFlow.py
        print "Warping..."
        remap[:, :, 0] = x + uv[:, :, 0]
        remap[:, :, 1] = y + uv[:, :, 1]
        remap = remap.astype('single')
        warped2 = cv2.remap(frame2, remap, None, cv2.INTER_CUBIC)
        # TODO#4: compute image gradient Ix, Iy, and Iz
        print "Computing Ix, Iy, Iz..."
        Ix = cv2.filter2D(warped2, -1, h, borderType = cv2.BORDER_CONSTANT)
        Iy = cv2.filter2D(warped2, -1, h.transpose(), borderType = cv2.BORDER_CONSTANT)
        Iz = warped2 - frame1


        # TODO#5: build linear system to solve HS flow
        # generate A,b for linear equation Ax = b
        # you may need use scipy.sparse.spdiags
        print "Buliding A, b..."
        U = np.reshape(uv[:, :, 0], (npixels, 1), order='F')
        V = np.reshape(uv[:, :, 1], (npixels, 1), order='F')

        row = []
        col = []
        data = []
        for p in xrange(npixels):
            px = p / H
            py = p % H
            add_nonzero_entry(row, col, data, p, p, Ix[py, px] * Ix[py, px] + lam * L[p, p])
            add_nonzero_entry(row, col, data, p + npixels, p + npixels, Iy[py, px] * Iy[py, px] + lam * L[p, p])
            add_nonzero_entry(row, col, data, p, p + npixels, Ix[py, px] * Iy[py, px])
            add_nonzero_entry(row, col, data, p + npixels, p, Ix[py, px] * Iy[py, px])
        nR, nC, nV = sp.find(L)
        for p in xrange(len(nR)):
            if nR[p] != nC[p]:
                add_nonzero_entry(row, col, data, nR[p], nC[p], lam * nV[p])
                add_nonzero_entry(row, col, data, nR[p] + npixels, nC[p] + npixels, lam * nV[p])

        A = coo_matrix((np.array(data), (np.array(row), np.array(col))), shape=(2 * npixels, 2 * npixels))
        U = np.reshape(uv[:, :, 0], (npixels, 1), order='F')
        V = np.reshape(uv[:, :, 1], (npixels, 1), order='F')
        term1 = L.dot(U)
        term2 = L.dot(V)
        data = []
        for p in xrange(npixels):
                px = p / H
                py = p % H
                data.append(- Ix[py, px] * Iz[py, px] - lam * term1[p, 0])
        for p in xrange(npixels):
                px = p / H
                py = p % H
                data.append(- Iy[py, px] * Iz[py, px] - lam * term2[p, 0])
        b = np.array(data)
        A = A.tocsc()
        ret = sslg.spsolve(A, b)
        deltauv = np.reshape(ret, uv.shape, order='F')
        # TODO#6:
        deltauv[deltauv == np.nan] = 0
        deltauv[deltauv > 1] = 1
        deltauv[deltauv < -1] = -1
        uv = uv + deltauv
        print "Median blurring..."
        uv[:, :, 0] = cv2.medianBlur(uv[:, :, 0].astype("float32"), 3)
        uv[:, :, 1] = cv2.medianBlur(uv[:, :, 1].astype("float32"), 3)
        #cv2.imshow('Estimate flow', ft.flowToColor(uv).astype('uint8'))
        #cv2.waitKey(0)
        print 'Warping step: %d, Incremental norm: %3.5f' %(i, np.linalg.norm(deltauv))
        # Output flow
    return uv


def estimateHSflow(frame1, frame2, lam = 80):
    H, W = frame1.shape

    # build the image pyramid
    pyramid_spacing = 1.0/0.8
    pyramid_levels = int(1 + np.floor(np.log(min(W, H) / 16.0) / np.log(pyramid_spacing * 1.0)))
    #pyramid_levels = 1
    smooth_sigma = np.sqrt(2.0)
    #  use cv2.GaussianBlur
    f = ft.fspecialGauss(2 * round(1.5 * smooth_sigma) + 1, smooth_sigma)

    pyramid1 = []
    pyramid2 = []

    pyramid1.append(frame1)
    pyramid2.append(frame2)
    for m in range(1, pyramid_levels):
        # TODO #1: build Gaussian pyramid for coarse-to-fine optical flow
        # estimation
        ph = int(np.ceil(pyramid1[-1].shape[0] * 0.8))
        pw = int(np.ceil(pyramid1[-1].shape[1] * 0.8))
        pyramid1[-1] = cv2.filter2D(pyramid1[-1], -1, f, borderType=cv2.BORDER_CONSTANT)
        pyramid1.append(cv2.resize(pyramid1[-1], (pw, ph), interpolation = cv2.INTER_CUBIC))
        pyramid2[-1] = cv2.filter2D(pyramid2[-1], -1, f, borderType=cv2.BORDER_CONSTANT)
        pyramid2.append(cv2.resize(pyramid2[-1], (pw, ph), interpolation = cv2.INTER_CUBIC))
    # coarst-to-fine compute the flow
    uv = np.zeros(((H, W, 2)))

    for levels in range(pyramid_levels - 1, -1, -1):
        print "level %d" % (levels)
        H1, W1 = pyramid1[levels].shape
        uv = ft.resample_flow(uv, H1, W1)
        uv = estimateHSflowlayer(pyramid1[levels], pyramid2[levels], uv, lam, 10)

    return uv
