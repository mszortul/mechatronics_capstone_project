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
from weighted_kde import *

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


def edf_plot(hist, title='test'):
    keys = []
    vals = []
    for kv in hist.items():
        keys.append(kv[0])
        vals.append(kv[1])
    plt.plot(keys, vals, 'b-')
    plt.xlabel('Degree')
    plt.ylabel('Magnitudes')
    plt.title(title)
    plt.show()

    
    
def est_kde(tan_list, mag_list, title='test', hist=False, ret=False, bw_method=None):
    x = np.linspace(-90, 90, 360)
    
    mag_arr = np.array(mag_list)
    tan_arr = np.array(tan_list)
    if hist is True:
        plt.hist(tan_arr, 90, (-90, 90), histtype='stepfilled', 
                 alpha=.2, normed=True, color='k', label='histogram', weights=mag_arr)
    pdf = gaussian_kde(tan_arr, weights=mag_arr, bw_method=bw_method)
    y = pdf(x)
    plt.plot(x, y, label='weighted_kde')
    plt.xlim((-90, 90))
    plt.xlabel('Degrees')
    plt.ylabel('Magnitudes')
    plt.title(title)
    plt.show()
    
    if ret:
        return y
    
    
    
def edf_comp(img, title='test', hist=False, ret=True, bw_method=None):
    gray = rgb2gray(img)
    gray_blur = cv2.GaussianBlur(gray, (5,5), 2)
    gx, gy = sobel(gray_blur)
    tans, mags = tan_mag(gx, gy)
    hist, tan_list, mag_list, tan_arr, mag_arr = edf(tans, mags)
    if ret:    
        kde_y = est_kde(tan_list, mag_list, title=title, hist=hist, ret=True, bw_method=None)
        return kde_y, tan_arr, mag_arr
    else:
        est_kde(tan_list, mag_list, title=title, hist=hist)
        
    #return hist, tan_list, mag_list
    
    
def get_maxima(kde_y):
    x = np.linspace(-90, 90, 360)
    maxm = argrelextrema(kde_y, np.greater)
    maxm = x[maxm]
    return maxm



def filter_maxima(maxm):
    valid_matches = []
    l = maxm.shape[0]
    fp = 0
    lp = l-1
    while (lp is not fp) and (lp > fp):
        diff = abs(maxm[fp] + maxm[lp])
        if diff < 15.0:
            valid_matches.append(fp)
            valid_matches.append(lp)
            fp += 1
            lp -= 1
        else:
            if abs(maxm[fp] > maxm[lp]):
                fp += 1
            else:
                lp -= 1
                
    return maxm[valid_matches]

def single_edge_image(tan_arr, mag_arr, alpha):
    h, w = tan_arr.shape
    single_line_image = np.zeros_like(mag_arr)
    for y in range(h):
        for x in range(w):
            if abs(tan_arr[y,x] - alpha) < 2.0:
                single_line_image[y,x] = mag_arr[y,x]
                
    return single_line_image


def smooth_line(arr, f_size):
    filt = np.ones(f_size) / f_size
    for i in range(int(f_size/2), arr.shape[0] - int(f_size/2)):
        new_val = sum(np.multiply(filt, arr[i-int(f_size/2):i+int(f_size/2)+1]))
        arr[i] = new_val
    
    return arr


def weighted_hough(edge_img, alpha, mag_arr):
    h, w = edge_img.shape
    acc_arr = np.zeros(int(math.sqrt(h**2 + w**2)))
    for y in range(h):
        for x in range(w):
            if edge_img[y,x] is not 0:
                a = alpha * 2 * (math.pi/180)
                rho = int(y * math.sin(a) + x * math.cos(a))
                acc_arr[rho] += mag_arr[y,x]
    
    return acc_arr

def smooth_line(arr, f_size):
    filt = np.ones(f_size) / f_size
    for i in range(int(f_size/2), arr.shape[0] - int(f_size/2)):
        new_val = sum(np.multiply(filt, arr[i-int(f_size/2):i+int(f_size/2)+1]))
        arr[i] = new_val
    
    return arr

def line(rho, alpha, h, w):
    x = np.arange(0, w, 1)
    line_im = np.zeros((h, w)).astype(np.uint8)
    for no in x:
        y = int(-x[no] / math.tan(alpha * (2*math.pi/180)) + (rho / (math.sin(alpha*(2*math.pi/180)))))
        line_im[y,no] = 255
    
    return line_im

