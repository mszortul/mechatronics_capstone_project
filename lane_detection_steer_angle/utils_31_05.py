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
from skimage.feature import peak_local_max
from operator import itemgetter
import copy
from numpy.linalg import inv
from math import atan, degrees, sin
from skimage import color
import pickle

from imports import *



'''

test = mpimg.imread('img/road.png')
gray = rgb2gray(test)
y, tan_arr, mag_arr = edf_comp(test)
edges, vm = get_edges_thetas(y, tan_arr, mag_arr)
gray_lines = draw_on_img(gray, edges, vm, mag_arr)

'''

EXTENSION = 20
YM = 550


def rad2deg(rad):
    '''
        Convert radian to degree

        Parameters:

            rad : scalar, radian value

        Return:

            deg : scalar, degree value

    '''

    deg = rad*(180/math.pi)
    return deg



def deg2rad(deg):
    '''
        Convert degree to radian

        Parameters:

            deg : scalar, degree value

        Return:

            rad : scalar, radian value

    '''

    rad = deg*(math.pi/180)
    return rad


def pre(im):
    '''
        Convert from RGB to grayscale, apply some blur k_size=(11,11), sigma=7

        Parameters:

            im : np.array, original RGB image

        Return:

            im1 : np.array, Blurred grayscale image

    '''

    im1 = rgb2gray(im)
    im1 = cv2.GaussianBlur(im1, (11, 11), 7)
    return im1


def rgb2gray(rgb):    
    '''
        Convert from RGB to grayscale (color.rgb2gray() recommended)

        Parameters:

            rgb : np.array, original RGB image

        Return:

            gray : np.array, grayscale image

    '''    
    gray = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    return gray

def gshow(img):
    '''
        Shortcut to display grayscale images

        Parameters:

            img : np.array, grayscale image

        Return:

            None

    '''
    
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()


def sobel(img):
    
    '''
        Apply sobel filter to a grayscale image

        Parameters:

            img : np.array, image to filter

        Return:

            gx : np.array, horizontal gradient image

            gy : np.array, vertical gradient image

    '''
    
    sh = np.array([[1,0,-1], 
                   [2,0,-2], 
                   [1,0,-1]], dtype=np.float)
    sv = np.array([[-1,-2,-1], 
                   [0,0,0], 
                   [1,2,1]], dtype=np.float)
    
    gx = cv2.filter2D(img, -1, sh).astype(np.float32)
    gy = cv2.filter2D(img, -1, sv).astype(np.float32)
    
    return (gx, gy)
    

def tan_mag(gx, gy):
    '''
        Get tangent and magnitude values
        
        Parameters:
            
            gx : np.array, horizontal gradient image

            gy : np.array, vertical gradient image
            
        Return:
        
            tans : np.array, tangent values of pixels (bunch of NaNs ahead!)
            
            mags : np.array, magnitude values of pixels
    
    '''
    
    h, w = gx.shape
    
    tans = np.zeros((h,w))
    mags = np.zeros((h,w))
    
    for y in range(h):
        for x in range(w):
            tan = rad2deg(math.atan(-gx[y,x] / gy[y,x]))
            tans[y,x] = tan
            mag = math.sqrt(gy[y,x]**2 + gx[y,x]**2)
            mags[y,x] = mag
    
    return tans, mags


def plot_vars(tans, mags):
    '''
        Get plot variables from tangent and magnitude values also filter tangent and magnitude values
        
        Parameters:
        
            tans : np.array, tangent values of pixels (bunch of NaNs ahead!)
            
            mags : np.array, magnitude values of pixels
            
        Return:
        
            hist : dict, 
                
                keys : orientations arange(-90,90,2) 
                values : magnitude weighted orientation voting results
                
            tan_arr : np.array, filtered tangent values (no NaNs!)
            
            mag_arr : np.array, magnitudes corresponding to NaN tangents are filtered
    
    '''
    
    tan_arr = np.zeros_like(tans)
    mag_arr = np.zeros_like(mags)
    h, w = tans.shape
    hist = {}
    xs = np.arange(-90, 91, 2)
    for x in xs:
        hist[x] = 0
    #tan_l = []
    #mag_l = []
    for y in range(h):
        for x in range(w):
            if (np.isnan(tans[y,x])):
                continue
            else:
                tan_arr[y,x] = tans[y,x]
                mag_arr[y,x] = mags[y,x]
                #tan_l.append(tans[y,x])
                #mag_l.append(mags[y,x])
                if (int(tans[y,x]) % 2 is 0):
                    hist[int(tans[y,x])] += mags[y,x]
                else:
                    hist[int(tans[y,x]) - 1] += mags[y,x]
    
    return hist, tan_arr, mag_arr


def edf_smooth(hist, sigma=5):
    '''
        Smooth the magnitude weighted histogram to clarify local maxima points
        
        Parameters:
        
            hist : dict, returned hist dictionary from plot_vars()
            
            sigma : scalar, sigma value of 1-dimensional gaussian filter
            
        Return:
        
            smoothx : list, x-axis values of smoothed edf histogram
            
            smoothy : list, y-axis values of smoothed edf histogram
    
    '''
    
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


def edf_comp(img, title='test', sigma=7, ret=True, prefunc=True, plott=False):
    '''
        Complete edf part of the algorithm, creates plot and returns needed objects
        
        Parameters:
        
            img : np.array, original RGB image
            
            title : string, title of plot
            
            sigma : scalar, sigma value of gaussian filtering of histogram
            
            ret : bool, assign False if you want just the plot part, otherwise leave it as it is
            
        Return:
        
            y : list, y-axis return of edf_smooth()
            
            tan_arr : np.array, return of plot_vars()
            
            mag_arr : np.array, return of plot_vars()
    
    '''
    if prefunc:
        gray_blur = pre(img)
    else:
        gray_blur = img
    gx, gy = sobel(gray_blur)
    tans, mags = tan_mag(gx, gy)
    hist, tan_arr, mag_arr = plot_vars(tans, mags)
    x, y = edf_smooth(hist, sigma=sigma)
    
    if plott:
        plt.plot(x, y, 'r-')
        plt.title('Kenar dağılım fonksiyonu oylama sonuçları')
        plt.xlabel('Oryantasyonlar')
        plt.ylabel('Büyüklükler toplamı')
        plt.show()
    
    if ret:
        return y, tan_arr, mag_arr

    
    
    
def center_maxima(maxm):
    '''
        Get maximas for center lane
        
        Parameters:
            
            maxm : list, returned list from filter_maxima()
            
        Return:
        
            center : list, center lane orientations (hopefully.. might be empty or with one element only)
    
    '''
    center = []
    l = len(maxm)
    if l:
        center.append(maxm[0])
    if l>1:
        center.append(maxm[-1])
        
    return center



def get_maxima(y):
    '''
        Get local maxima points of smoothed edf
        
        Parameters:
        
            y : list, y-axis return of edf_comp()
            
        Return:
        
            maxm : list, local maxima points, angles between x-axis and detected lines (-90, 90)
    
    '''
    
    x = np.linspace(-90, 90, 360)
    maxm = argrelextrema(y, np.greater)
    maxm = x[maxm]
    return maxm

def filter_maxima(maxm):
    '''
        Apply a threshold to eliminate false positive local maxima angles
        
        Parameters:
        
            maxm : list, returned list of get_maxima()
            
        Return:
        
            vm : list, true angles (hopefully..)
    
    '''
    
    valid_matches = []
    l = maxm.shape[0]
    fp = 0
    lp = l-1
    while (lp is not fp) and (lp > fp):
        diff = abs(maxm[fp] + maxm[lp])
        if diff < 20.0:
            valid_matches.append(fp)
            valid_matches.append(lp)
            fp += 1
            lp -= 1
        else:
            if abs(maxm[fp] > maxm[lp]):
                fp += 1
            else:
                lp -= 1
    
    vm = maxm[valid_matches]
    
    return vm

def single_edge_image(tan_arr, mag_arr, alpha):
    '''
        Get single edge magnitudes image
        
        Parameters:
        
            tan_arr : np.array, return of edf_comp()
            
            mag_arr : np.array, return of edf_comp()
            
            alpha : scalar, angle of desired line
            
        Return:
        
            single_edge_image : np.array, single edge magnitude image
    
    '''
    
    h, w = tan_arr.shape
    single_edge_image = np.zeros_like(mag_arr)
    for y in range(h):
        for x in range(w):
            if abs(tan_arr[y,x] - alpha) < 2.0:
                single_edge_image[y,x] = mag_arr[y,x]
                
    return single_edge_image


def get_edges_thetas(y, tan_arr, mag_arr):
    '''
        Get edge images and corresponding angles
        
        Parameters:
        
            y : list, Return of edf_comp()
            
            tan_arr : np.array, Return of edf_comp()
            
            mag_arr : np.array, Return of edf_comp()
            
        Return:
        
            edges : list, every element is a edge image
            
            vm : list, angles of corresponding edges found
    
    '''
    
    maxm = np.sort(get_maxima(y))
    #print(maxm)
    vm = np.sort(filter_maxima(maxm))
    print('Lokal maksimumlar:', vm)
    #print(vm)
    center_vm = center_maxima(vm)
    #print(center_vm)
    edges = []
    for i in range(len(center_vm)):
        edges.append(single_edge_image(tan_arr, mag_arr, center_vm[i]))
    return edges, center_vm


def vm_to_hough(vm):
    '''
        Result of edf maximas and hough transform angles are lightly different, this function fixes that problem
        
        Parameters:
        
            vm : list, Return of get_edges_thetas()
            
        Return:
        
            ret : list, same length as input, just modified for next phase
    
    '''
    
    ret = np.zeros_like(vm)
    for i in range(len(vm)):
        if vm[i] < 0:
            ret[i] = -90 - vm[i]
        else:
            ret[i] = 90 - vm[i]
    return ret


def weighted_hough(edge_img, theta, mag_arr):
    '''
        Apply weighted hough transform to a single image
        
        Parameters:
        
            edge_img : np.array, magnitude image of one edge
            
            theta : scalar, angle about to be searched in edge_img
            
            mag_arr : np.array, magnitudes are weights
            
        Return:
        
            acc : np.array, accumulator array of hough transform (one dimensional)
            
            diag : scalar, diagonal length of edge image
            
            theta : scalar, same theta of input
    
    '''
    
    t = deg2rad(theta)
    h, w = edge_img.shape
    diag = int(math.ceil(math.sqrt(h**2 + w**2)))
    acc = np.zeros(2*diag+1)
    for y in range(h):
        for x in range(w):
            if edge_img[y,x] > 0:
                rho = int((y * math.sin(t) + x * math.cos(t)) + diag)
                acc[rho] += mag_arr[y,x]
                
    return acc, diag, theta


def calc_p(points):
    '''
        Calculate extension point of line drawing
        
        Parameters:
        
            points : tuple of two tuples, ((x1,y1),(x2,y2))
            
        Return:
        
            n_points : tuple of two tuples, ((x1,y1),(x3,y3))
    
    '''
    
    x1, y1 = points[0]
    x2, y2 = points[1]
    
    d = int(EXTENSION * (x2-x1) / (y1 - y2))
    
    x3 = x2 + d
    y1 = y1 + EXTENSION
    y3 = 0
    
    n_points = ((x1,y1), (x3, y3))
    
    return n_points


def line(rho, alpha, h, w, thicc=5, ext=True):
    '''
        Draw a line with given parameters
        
        Parameters:
        
            rho : scalar, minimum edge distance from pixel(0,0)
            
            alpha : scalar, hough line orientation
            
            h : scalar, line image height
            
            w : scalar, line image width
            
            thicc : scalar, line thickness
            
        Return:
        
            line_im : np.array, drawn line image
    
    '''
    if ext:
        line_im = np.zeros((h+EXTENSION, w)).astype(np.uint8)
    else:
        line_im = np.zeros((h, w)).astype(np.uint8)
    p = []
    for x in range(w):
        y = int( -x / math.tan(alpha * (math.pi/180)) + (rho / (math.sin(alpha*(math.pi/180)))))
        if (y < h and y >= 0):
            p.append((x, y))
    
    p.sort(key=itemgetter(1))
    last = len(p) - 1
    #print('orig', p[0], p[last])
    
    ps = calc_p((p[last], p[0]))
    
    #print('new', ps)
    cv2.line(line_im, ps[0], ps[1], (255,255,255), thickness=thicc)
    return line_im

def draw_on_img(gray, edges, vm, mag_arr, y1=470, y2=650, x1=None, x2=None, thicc=20):
    '''
    Draw edges on grayscale and get lane boundry region of interest
    
    Parameters:
        
        gray : np.array, grayscale of original image
        
        edges : list, return of get_edges_thetas()
        
        vm : list, return of get_edges_thetas()
        
        mag_arr : np.array, return of edf_comp()
        
        y1 : scalar, search region limit (lower y)
        
        y2 : scalar, search region limit (greater y)
        
        x1 : scalar, search region limit (lower x)
        
        x2 : scalar, search region limit (greater x)
        
        thicc : scalar, line thickness
        
    Return:
    
        gray_lines : np.array, line drawn grayscale image
        
        lbrois : list
        
            [lbroi1, lbroi2, ..., lbroiN]
            
            lbroiN : Nth edge area for next phase of algorithm
    
    '''
    ang_buf = [deque([]), deque([])]
    for i in range(len(vm)):
        ang_buf[i].append(vm[i])
    gray_lines = np.copy(gray)
    h, w = mag_arr.shape
    h_degs = vm_to_hough(vm)
    lbrois = []
    for i in range(len(edges)):
        lbroi = np.zeros_like(gray)
        edge_im = edges[i]
        theta = h_degs[i]
        acc, diag, theta = weighted_hough(edge_im, theta, mag_arr)
        smooth_acc = gaussian_filter1d(acc, sigma=51)
        x = np.arange(0, 2*diag+1, 1)
        maxm = argrelextrema(smooth_acc, np.greater)
        maxm = x[maxm]
        
        points = []
        for maxima in maxm:
            points.append((maxima, smooth_acc[maxima]))
        points.sort(key=itemgetter(1))
        
        rho = points[-1][0] - diag
        
        line_img = line(rho, theta, h, w, thicc=thicc)
        
        coors = np.zeros(4).astype(np.uint32)
        
        coors[0] = 0 if y1 is None else y1-EXTENSION
        coors[1] = h if y2 is None else y2
        coors[2] = 0 if x1 is None else x1
        coors[3] = w if x2 is None else x2
        
        #for g in range(4):
            #coors[g] = int(coors[g])
            #print(coors[g])
        lbroi[coors[0]:coors[1], coors[2]:coors[3]] += line_img
        lbrois.append(lbroi)
        gray_lines[coors[0]:coors[1], coors[2]:coors[3]] += line_img
    
    return gray_lines, lbrois, ang_buf
