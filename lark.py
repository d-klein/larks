""" 
Implements the Linear Adaptive Regression Kernel (LARK)
descriptor as defined in 

Hae Jong Seo; Milanfar, P., "Face Verification Using the LARK Representation,
" Information Forensics and Security, IEEE Transactions on , 
vol.6, no.4, pp.1275,1286, Dec. 2011
doi: 10.1109/TIFS.2011.2159205
"""

__author__ = "D. Klein"
__license__ = "GPL 2"
__version__ = "1.0"

import numpy as np
from scipy import ndimage
from scipy import misc
import matplotlib.pyplot as plt
import math
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)

def _unblock(arr):
    """
    Given an numpy array of the shape (x,y,z,w)
    this function merges vstacks and hstacks
    all subarrays (z,w) to form an array
    of size (x*z, y*w).

    Example:
    
    An array of shape (2,2,2,2):

    1  2    5  6    
    3  4    7  8

    9  10  13 14
    11 12  15 16 

    will be merged in the obvious way to dimension (4,4):

    >>> import numpy as np
    >>> a = np.array(((((1,2),(3,4)),((5,6),(7,8))),(((9,10),(11,12)),((13,14),(15,16)))))
    >>> a.shape 
    (2, 2, 2, 2)
    >>> a
    array([[[[ 1,  2],
            [ 3,  4]],

           [[ 5,  6],
            [ 7,  8]]],


          [[[ 9, 10],
            [11, 12]],

           [[13, 14],
            [15, 16]]]])
    
    >>> unblock(a)
    array([[ 1,  2,  5,  6],
           [ 3,  4,  7,  8],
           [ 9, 10, 13, 14],
           [11, 12, 15, 16]])
    >>> unblock(a).shape
    (4,4)
    >>> a = np.array(((((1,2),),((5,6),)),(((9,10),),((13,14),))))
    >>> lark.unblock(a)
    array([[ 1,  2,  5,  6],
           [ 9, 10, 13, 14]])

    >>> a = np.array(((((1,),(3,)),((5,),(7,))),(((9,),(11,)),((13,),(15,)))))
    >>> lark.unblock(a)
    array([[ 1,  5],
           [ 3,  7],
           [ 9, 13],
           [11, 15]])
    """
    x,y,_,_ = arr.shape
    reshaped = []
    for i in xrange(0,x):
        if y >= 2:
            reshaped_i = np.hstack((arr[i][0],arr[i][1]))
        else:
            reshaped_i = arr[i][0]
        for j in xrange(2,y):
            reshaped_i = np.hstack((reshaped_i,arr[i][j]))
        reshaped.append(reshaped_i)
    reshaped = np.array(reshaped)
    if x >= 2:
        res = np.vstack((reshaped[0],reshaped[1]))
        for i in xrange(2,x):
            res = np.vstack((res,reshaped[i]))
        return res
    else:
        return reshaped[0]

def _fspecial(f_type, arg1 = None):
    """
    implements the 'fspecial' command of
    octave-forge which

    creates spatial filters for image processing.
    f_type determines the shape of the filter. Currently
    only "disk" is supported:

    "disk": 
    Circular averaging filter. The optional argument arg1 controls the 
    radius of the filter. If arg1 is an integer R, a 2 R + 1 filter 
    is created. By default a radius of 5 is used. If the returned matrix 
    corresponds to a cartesian grid, each element of the matrix is weighted 
    by how much of the corresponding grid square is covered by a disk of 
    radius R and centered at the middle of the element R+1,R+1. 
    """
    if f_type == 'disk':
        radius = 5.0
        if not arg1 == None:
            if isinstance(arg1, float):
                radius = arg1
            elif isinstance(arg1, int):
                radius = float(arg1)
            else:
                raise ValueError("arg1 must be int or float variable")
        [x,y] = np.meshgrid(np.arange(-radius,radius+1), np.arange(-radius,radius+1))
        r = np.sqrt(x**2 + y**2)        
        f = (r <= radius).astype(float)
        f = f / np.sum( f )
        return f
    else:
        raise ValueError("not implemented yet")


def _gray2real(img):
    x = img.shape[0]
    y = img.shape[1]
    fxy = np.zeros((x,y),dtype=np.float_)
    for i in xrange(0,x):
        for j in xrange(0,y):
            fxy[i,j] = np.float_(img[i][j]) / 255.0
    return fxy

def _raw_larks(Inimg, wsize=5, interval=3, h=2, alpha=0.13):
    """
    same as larks(...), but returns an (V,M,N) structure
    where M,N is the dimension of the image, and V
    the size of a LARK (i.e. V = wsize*wsize)
    """
    inimg = _gray2real(Inimg)
    inimg_std = inimg/np.std(inimg,ddof=1)
    LARK = _make_lark(inimg_std,wsize,h,interval,alpha)
    return LARK

def larks(Inimg, wsize=5, interval=3, h=2, alpha=0.13):
    """
    Inimg must be a gray-level image normalized between 0 and 255
    wsize is the window size of a LARK, e.g. wsize = 5 takes a 5x5 LARK window 
    interval defines the the covariance matrix to be computex interval-pixels apart
    h is a smoothing parameter 
    alpha is a sensitivity parameter between 0.0 and 1.0 

    computes K = [k_1, ..., k_n] \in R^(P \times n)
    where k_i is a vectorized version of a lark \emph{K}, 
    and n is the number of LARKS in the image.

    P will be the squared interval size

    This is the function that should be called when
    
    Example:
    >>> import lark
    >>> from skimage import data, color
    >>> img = data.lena()
    >>> img_gray = color.rgb2gray(img)
    >>> img_lrks = lark.larks(img_gray)
    """
    inimg = _gray2real(Inimg)
    inimg_std = inimg/np.std(inimg,ddof=1)
    LARK = _make_lark(inimg_std,wsize,h,interval,alpha)
    x,y,z = LARK.T.shape
    return LARK.T.reshape(x*y,z,order='F').copy()


def _make_lark(img,wsize,h,interval,alpha):
    win = (wsize-1)/2
    zx,zy = np.gradient(img.T)
    zxp = np.lib.pad(zx,(win,win),mode='symmetric').T
    zyp = np.lib.pad(zy,(win,win),mode='symmetric').T
    M = img.shape[0]
    N = img.shape[1]
    C11 = np.zeros((M,N))
    C12 = np.zeros((M,N))
    C22 = np.zeros((M,N))
    tmp = np.zeros((2,2))
    G = np.zeros((wsize*wsize,2))
    gx = np.zeros((wsize,wsize))
    gy = np.zeros((wsize,wsize))
    K = _fspecial('disk',win)
    K = K / K[win, win]
    le = np.sum(K[:])
    for i in xrange(1,M+1,interval):
        for j in xrange(1,N+1,interval):
            gx = (zxp[i-1:i+wsize-1][:,j-1:j+wsize-1])*K
            gy = (zyp[i-1:i+wsize-1][:,j-1:j+wsize-1])*K
            G = np.array([gx.flatten(1),gy.flatten(1)]).T
            u, s, v = np.linalg.svd(G, full_matrices=False)
            S1 = (s[0]+1.0) / (s[1]+1.0)
            S2 = 1.0/S1
            m1 = (S1 * v[:,0])
            m2 = v[:,0]
            m3 = np.dot(np.array([m1]).T,np.array([m2]))
            m4 = (S2 * v[:,1])
            m5 = v[:,1]
            m6 = np.dot(np.array([m4]).T,np.array([m5]))
            m7 = ((s[0] * s[1] + 0.0000001)/le)**alpha
            tmp = (m3 + m6) * m7
            C11[i-1][j-1] = tmp[0][0]
            C12[i-1][j-1] = tmp[1][0]
            C22[i-1][j-1] = tmp[1][1]

    C11 = C11[0::interval][:,0::interval]
    C12 = C12[0::interval][:,0::interval]
    C22 = C22[0::interval][:,0::interval]

    M,N = C11.shape
    C11 = np.lib.pad(C11,(win,win),mode='symmetric')
    C12 = np.lib.pad(C12,(win,win),mode='symmetric')
    C22 = np.lib.pad(C22,(win,win),mode='symmetric')
    
    x2,x1 = np.meshgrid(range(-win,win+1),range(-win,win+1))
    x12 = np.dot(2,x1) * x2
    x11 = x1**2
    x22 = x2**2
    x1x1 = (np.tile(x11.reshape(1,wsize**2,order='F').copy(),(M*N,1))).T.reshape((wsize,wsize,M,N))
    x1x2 = (np.tile(x12.reshape(1,wsize**2,order='F').copy(),(M*N,1))).T.reshape((wsize,wsize,M,N))
    x2x2 = (np.tile(x22.reshape(1,wsize**2,order='F').copy(),(M*N,1))).T.reshape((wsize,wsize,M,N))
    LARK = np.zeros((wsize,wsize,M,N))
    for i in xrange(0,wsize):
        for j in xrange(0,wsize):
            LARK[j][i] = C11[i:i+M][:,j:j+N] * x1x1[j][i] + C12[i:i+M][:,j:j+N] *x1x2[j][i] + C22[i:i+M][:,j:j+N] * x2x2[j][i]
    LARK = np.exp(-(LARK)/h)
    LARK = LARK.reshape((wsize**2,M,N),order='C').copy()
    LARK = LARK / np.tile((np.sum(LARK.T,axis=2)).T, (wsize ** 2,1,1))
    return LARK

def _make_visual_lark(larks, M, N, wsize_skr):
    win_skr = (wsize_skr-1) // 2
    out_larks = [[[] for i in xrange(0,N-wsize_skr+2,wsize_skr)] for i in range(0,M-wsize_skr+2,wsize_skr)]
    ii = 0
    for i in xrange(0,M-wsize_skr+1,wsize_skr):
        jj = 0
        for j in xrange(0,N-wsize_skr+1,wsize_skr):
            out_larks[ii][jj] = np.array(larks.T[j+win_skr][i+win_skr].reshape(wsize_skr,wsize_skr,order='F').copy())
            jj = jj+1
        ii = ii+1
    ol = np.array(out_larks)
    ol1 = _unblock(ol)
    return ol1

def visual_lark(Inimg, wsize=5, interval=3, h=2, alpha=0.13):
    """
    computes larks, and then removes all overlapping patches
    will hence give a visual impression of the generated larks
    and can be used to plot an image. Useful when playing
    around with parameters.

    Example
    >>> import lark
    >>> from skimage import data, color
    >>> img = data.lena()
    >>> img_gray = color.rgb2gray(img)
    >>> import matplotlib.pyplot as plt
    >>> img_vlrks = lark.visual_lark(img_gray)
    >>> 
    >>> plt.imshow(img_vlrks)
    >>> plt.colorbar()
    >>> plt.show()

    >>> img_vlrks = lark.visual_lark(img_gray,3,3,2,0.13)
    >>> plt.imshow(img_vlrks,interpolation='none')
    >>> plt.colorbar()
    >>> plt.show()

    >>> img_vlrks = lark.visual_lark(img_gray,3,2,10,0.2)
    >>> plt.imshow(img_vlrks,interpolation='none')
    >>> plt.colorbar()
    >>> plt.show()

    """
    lrks = _raw_larks(Inimg,wsize,interval,h, alpha)
    _,M,N = lrks.shape
    return _make_visual_lark(lrks, M,N,wsize)




"""
experiments for implementation
to be removed later


im = misc.imread("1a_sw1.png")


lrks = _raw_larks(im,5,3,2,0.13)
#lrks = raw_larks(im)
lrks1 = larks(im,5,3,2,0.13)
#print lrks
print lrks.shape
#out_larks = draw_non_overlapping(lrks,120,87,5)
out_larks = visual_lark(im)
print out_larks.shape
#print larks


#misc.imsave("foo.png",out_larks)

#plt.imshow(out_larks, interpolation='none')
plt.imshow(out_larks)
plt.colorbar()
plt.show()
#ax.imshow(data, extent=[0, 1, 0, 1])
#print (fspecial('disk'))
"""
