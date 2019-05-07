import numpy as np
import cv2
from numba import jit
import numba

@jit
def threshold_otsu_histogram(histo: numba.uint32):
    """
    Thresholds a histogram vector
    Implemenation Notes: Not vectorized optimized, use numba.jit
    adapted from:
    https://stackoverflow.com/questions/33041900/opencv-threshold-with-mask"""
    FLT_EPSILON = np.finfo(float).eps
    M = sum(histo)
    scale = 1. / (M)
    mu = scale * (np.arange(len(histo))*histo).sum()

    mu1 = 0
    q1 = 0;
    max_sigma = 0
    max_val = 0;

    histo_sc = histo * scale
    #histo_sc_cum = np.cumsum(histo_sc)
    #histo_sc_cum_inv = 1 - histo_sc_cum
    for i in range(0, len(histo)):
        p_i = histo_sc[i]
        p_i = histo[i] * scale
        mu1 *= q1
        q1 += p_i
        q2 = 1. - q1
        #q1 = histo_sc_cum[i]
        #q2 = histo_sc_cum_inv[i]
        if (min(q1, q2) < FLT_EPSILON) or (max(q1, q2) > 1. - FLT_EPSILON):
            continue
        mu1 = (mu1 + i*p_i) / q1
        mu2 = (mu - q1*mu1) / q2
        
        sigma = q1*q2*(mu1 - mu2)*(mu1 - mu2)

        if (sigma > max_sigma):
            max_sigma = sigma;
            max_val = i;
    return max_val


def crop(bbox, im):
    return im[bbox[1]:bbox[1]+bbox[3], 
              bbox[0]:bbox[0]+bbox[2]]

def crop_roi(bbox, contour):
    contour = np.asarray(contour)
    return contour - np.r_[bbox[0], bbox[1]]

def circle(ker_size = 5):
    'draw a circle of given diameter'
    assert ker_size % 2 == 1, 'provide an odd integer'
    radius = ker_size//2
    x = np.arange(ker_size, dtype=np.uint8) - radius 
    mask = (x[:, np.newaxis]**2 + x[np.newaxis]**2) <= radius**2
    return mask

def intersection_over_union(x, y, axis=None, epsilon=1e-6):
    union = np.logical_or(x, y)
    intersection = np.logical_and(x, y)
    return (intersection.sum(axis) + epsilon) / (union.sum(axis) + epsilon)

def masked_otsu(image, mask):
    """computes a thresholding mask on a masked image"""
    image_masked = image[mask]
    histo = np.bincount(image_masked)
    thr = threshold_otsu_histogram(histo)
    mask_thr_ventr = (~(((image<thr) * mask)  | (~mask)))
    return mask_thr_ventr
    

def find_longest_contour(mask_thr_ventr, squeeze=True):
    # extract contours from the mask
    verts, _ = cv2.findContours((mask_thr_ventr).astype(np.uint8),
                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    verts = verts[np.argmax([len(x) for x in verts])]
    return verts[:,0,:]

def threshold_i_contour(sample,
                        masked_threshold_fn=masked_otsu,
                        prefilter_fn = lambda x: x,
                        mode = 'close',
                        ker_size=5,
                        redraw=True):
    """thresholds an image to derive an i-mask and an i-contour 
    given an o-mask
    :param  masked_threshold_fn: thresholding function that takes (image, mask)
    :param  prefilter_fn:        a function to pre-filter the image (default: identity)
    :param  mode:                morphologic operation (close, open, dilate, erode) or 
                                 median filter to be applied to the thresholded image
    :param ker_size:             kernel size for `mode` morphologic / filtering operation
    :param redraw:               redraw the contour (helps to remove small disconnected areas

    :OUTPUT {'icontour' : [[x,y], ...] , 'imask': np.ndarray(..., dtype=bool)}
    """
    
    mask_thr_ventr = masked_otsu(prefilter_fn(sample['image']),
                                 sample['omask']).astype(np.uint8)

    # perform morphologic closure to smooth the mask
    if ker_size>1:
        attr = f'MORPH_{mode.upper()}'
        if hasattr(cv2, attr):
            ker = circle(ker_size).astype(np.uint8)
            mask_thr_ventr = cv2.morphologyEx(mask_thr_ventr, getattr(cv2, attr), ker)
        elif mode.lower() == 'median':
            mask_thr_ventr = cv2.medianBlur(mask_thr_ventr, ker_size)
        else:
            raise ValueError(f'unknown mode: {mode}')
    
    # extract contours from the mask
    verts = find_longest_contour(mask_thr_ventr, squeeze=False)
    
    if redraw:
        mask_thr_ventr = np.zeros_like(mask_thr_ventr)
        mask_thr_ventr = cv2.drawContours(mask_thr_ventr, [verts], 0, 1, -1)
    
    verts = verts[:,0,:]
    return {'imask':mask_thr_ventr.astype(bool), 'icontour':verts}
