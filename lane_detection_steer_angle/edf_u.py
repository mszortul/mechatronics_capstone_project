import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from scipy import ndimage
from scipy import signal
from scipy import misc
import math
import cv2
from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter1d


# Insert grayscale image

def sobel(img):
    sh = np.array([[1,0,-1], 
                   [2,0,-2], 
                   [1,0,-1]], dtype=np.float)
    sv = np.array([[-1,-2,-1], 
                   [0,0,0], 
                   [1,2,1]], dtype=np.float)
    
    gx = cv2.filter2D(img, -1, sh).astype(np.float32)
    gy = cv2.filter2D(img, -1, sv).astype(np.float32)
    
    return (gx, gy)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gshow(img):
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
    
    
def tan_mag(gx, gy):
    h, w = gx.shape
    tans = np.zeros_like(gx)
    mags = np.zeros_like(gy)
    for y in range(h):
        for x in range(w):
            mags[y,x] = np.sqrt(np.power(gx[y,x],2) + np.power(gy[y,x],2))
            tans[y,x] = -math.atan(gy[y,x]/gx[y,x]) * (180/math.pi)
    return tans, mags


def tan_modf(tan_val):
    if tan_val >= 0:
        return 90-tan_val
    else:
        return -90-tan_val

def edf(tans, mags):
    tan_list = []
    mag_list = []
    tan_arr = np.zeros_like(tans)
    mag_arr = np.zeros_like(mags)
    h, w = tans.shape
    keys = np.arange(-90, 91, 2)
    hist = {}
    for key in keys:
        hist[key] = 0
    for y in range(h):
        for x in range(w):
            if (np.isnan(tans[y,x])):
                continue
            else:
                tans[y,x] = tan_modf(tans[y,x])/2
                tan_arr[y,x] = tans[y,x]
                mag_arr[y,x] = mags[y,x]
                mag_list.append(mags[y,x])
                tan_list.append(tans[y,x])
                if (int(tans[y,x]) % 2 is 0):
                    hist[int(tans[y,x])] += mags[y,x]
                else:
                    hist[int(tans[y,x]) - 1] += mags[y,x]
            
    return hist, tan_list, mag_list, tan_arr, mag_arr

def edf_smooth(hist, sigma=5):
    k = []
    v = []
    for co in hist.items():
        k.append(co[0])
        v.append(co[1])
    new_k = []
    c = -90
    while c <= 90:
        new_k.append(c)
        c += 0.5
    new_v = []
    for i in range(90):
        diff = v[i+1] - v[i]
        step = diff/4
        for j in range(4):
            new_v.append(v[i] + step * j)
    new_v.append(0.0)
    smoothy = gaussian_filter1d(new_v, sigma)
    smoothx = new_k
    
    return smoothx, smoothy


def edf_comp(img, title='test', sigma=5, ret=True):
    gray = rgb2gray(img)
    gray_blur = cv2.GaussianBlur(gray, (5,5), 2)
    gx, gy = sobel(gray_blur)
    tans, mags = tan_mag(gx, gy)
    hist, tan_list, mag_list, tan_arr, mag_arr = edf(tans, mags)
    x, y = edf_smooth(hist, sigma=sigma)
    plt.plot(x, y)
    plt.show()
    if ret:
        return y, tan_arr, mag_arr
    
 