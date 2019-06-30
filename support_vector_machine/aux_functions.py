import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
import random
from skimage.feature import hog
import time
import pickle

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn import decomposition

from moviepy.editor import VideoFileClip, ImageSequenceClip

from scipy.ndimage.measurements import label

# vehicle detection parameters
#heatmap_arr = []
#heatmap_filterSize = 8
#cars_ar = []
#frame_i = 0


# parameters for hog
orient = 9
pixels = (8,8)
cell_size = (2,2)
hog_ch = [0,1,2]



#------------------------------------------------------------------------------------------------------#



class Car:
    def __init__(self):
        self.bboxAr = []
        self.bboxFilter = 6
        self.failedDetectCount = 0
        self.failedDetectThresh = 2
        self.curBboxArea = 0

    def bboxSize(self, bbox):
        xSize = bbox[1][0] - bbox[0][0]
        ySize = bbox[1][1] - bbox[0][1]
        return xSize*ySize

    def updatePos(self, bbox):
        if bbox == None:
            self.failedDetectCount += 1
            if self.failedDetectCount > self.failedDetectThresh:
                self.bboxAr = []
        else:
            self.failedDetectCount = 0
            # check if current position is much different
            if len(self.bboxAr):
                if (abs(bbox[0][0]-np.mean(self.bboxAr, axis=0).astype(int)[0][0])) > 100 or (abs(bbox[1][0]-np.mean(self.bboxAr, axis=0).astype(int)[1][0]) > 100):
                    self.bboxAr = []
            self.bboxAr.append(bbox)
            if len(self.bboxAr) > self.bboxFilter:
                self.bboxAr = self.bboxAr[1:]

    def getBbox(self):
        if self.bboxAr != []:
            # smooth bbox
            bbox = np.mean(self.bboxAr, axis=0).astype(int)
            self.curBboxArea = self.bboxSize(bbox)
            return bbox
            #return np.average(self.bboxAr)
        else:
            return None




#------------------------------------------------------------------------------------------------------#




# a function to convert color space from BGR to cspace

# cspace = RGB, HSV, LUV, HLS, YUV, YCrCb, GRAY

def convert_color(img, cspace):
    image = np.copy(img)
    
    constant_dict = {'RGB':cv2.COLOR_BGR2RGB, 
                     'HSV':cv2.COLOR_BGR2HSV, 
                     'LUV':cv2.COLOR_BGR2LUV,
                     'HLS':cv2.COLOR_BGR2HLS,
                     'YUV':cv2.COLOR_BGR2YUV,
                     'YCrCb':cv2.COLOR_BGR2YCR_CB,
                     'GRAY':cv2.COLOR_BGR2GRAY, 
                     'RGB2YCrCb':cv2.COLOR_RGB2YCR_CB, 
                     'BGR':cv2.COLOR_RGB2BGR, 
                     'RGB2GRAY':cv2.COLOR_RGB2GRAY}
    
    converted = cv2.cvtColor(image, constant_dict[cspace])
    
    return converted


#------------------------------------------------------------------------------------------------------#


# short version of sklearn.feature.hog()
def hog_short(img, vis=False, feature_vector=True):
    if vis is True:
        features, hog_image = hog(img, orientations=9, cells_per_block=(2,2), 
                                  pixels_per_cell=(8,8), transform_sqrt=False, 
                                  visualise=True, feature_vector=feature_vector)
        return features, hog_image
    
    else:
        features = hog(img, orientations=9, cells_per_block=(2,2), 
                       pixels_per_cell=(8,8), transform_sqrt=False,
                       visualise=False, feature_vector=feature_vector)
        return features
    

#------------------------------------------------------------------------------------------------------#
    
    
    
# a function to iterate through training examples
 
# it takes pathnames as a list and returns hog feature vectors also as a list
# hog_channel = 0,1,2 or ALL

def hog_from_file(pathnames, hog_channel=[0,1,2], cspace='YCrCb', feature_vector=True):
    # Create a list to append feature vectors to
    all_features = []
    
    for image_path in pathnames:
        # read images one by one
        image = cv2.imread(image_path)
        # convert color space to desired one
        feat_img = convert_color(image, cspace)
            
        # extract features from single image
        
        features = []

        for channel in hog_channel:
            features.append(hog_short(feat_img[:,:,channel], feature_vector=feature_vector))

        features = np.ravel(features)

        # append features to return list
        all_features.append(np.concatenate((features)))
        
    return all_features


#------------------------------------------------------------------------------------------------------#



# takes pathnames of images
def hog_from_file_gray(pathnames, feature_vector=True, gaussian_blur=False):
    all_features = []
    exec_str = ''
    
    if gaussian_blur is True:
        exec_str = 'feat_img = cv2.GaussianBlur(feat_img, ksize=(5,5), sigmaX=0)'
        
    
    
    for image_path in pathnames:
        # read images one by one
        image = cv2.imread(image_path)
        feat_img = convert_color(image, 'GRAY')
        
        exec(exec_str)
        
        features = hog_short(feat_img)

        features = np.ravel(features)

        # append features to return list
        all_features.append(features)
        
    return all_features



#------------------------------------------------------------------------------------------------------#



# takes images as a np.array / list of np.array's
def hog_from_list(imgs, hog_channel=[0,1,2], cspace='YCrCb', feature_vector=True):
    all_features = []
    
    for img in imgs:
        image = np.copy(img)
       
        # convert color space to desired one
        feat_img = convert_color(image, cspace)
        
        # extract features from single image
        features = []

        for channel in hog_channel:
            features.append(hog_short(feat_img[:,:,channel], feature_vector=feature_vector))

        features = np.ravel(features)

        # append features to return list
        all_features.append(features)
        
    return all_features


#------------------------------------------------------------------------------------------------------#



# takes images as a np.array / list of np.array's
def hog_from_list_gray(imgs, feature_vector=True, gaussian_blur=False):
    all_features = []
    exec_str = ''
    
    if gaussian_blur is True:
        exec_str = 'feat_img = cv2.GaussianBlur(feat_img, ksize=(5,5), sigmaX=0)'
    
    for img in imgs:
        feat_img = np.copy(img)
        
        exec(exec_str)
        
        features = hog_short(feat_img)

        features = np.ravel(features)

        # append features to return list
        all_features.append(features)
        
    return all_features


#------------------------------------------------------------------------------------------------------#



# returns list of possible windows to be searched on image
def slide_window(img, x_start_stop=[0, None], y_start_stop=[0, None], 
                    windowSizeArr=[64], xy_overlap=(0.5, 0.5)):
    
    # If x and/or y stop positions not defined, set to image size
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    
    # Initialize a list to append window positions to
    window_list = []
    for windowSize in windowSizeArr:
        xy_window = (windowSize, windowSize)
        
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        
        # Compute the number of windows in x/y
        nx_windows = np.int(xspan / nx_pix_per_step) - 1
        ny_windows = np.int(yspan / ny_pix_per_step) - 1
        
        # Loop through finding x and y window positions
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]

                # Append window position to list
                window_list.append(((int(startx), int(starty)), (int(endx), int(endy))))
    
    # Return the list of windows
    return window_list



#------------------------------------------------------------------------------------------------------#



# crops given windows positions from image, return as a list
def get_window_imgs(img, windows, outSize=64, resize=True):
    imgs = []
    for window in windows:
        if resize:
            imgs.append(cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64,64)))
        else:
            imgs.append(img[window[0][1]:window[1][1], window[0][0]:window[1][0]])
    imgs = np.array(imgs)
    return imgs



#------------------------------------------------------------------------------------------------------#



# takes colored (0,0,0) copy of image, adds heat(color intensity) to pixels in box_list
def add_heat(heatmap, box_list):
    # Iterate through list of boxes
    for box in box_list:
        
        # Add += 1 for all pixels inside each box
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap



#------------------------------------------------------------------------------------------------------#



# applies threshold for eliminating false positive results
def apply_threshold(heatmap, threshold):
    # copy image
    th_heatmap = np.copy(heatmap)
    
    # Zero out pixels below the threshold
    th_heatmap[heatmap <= threshold] = 0
    
    # Return thresholded map
    return th_heatmap


#------------------------------------------------------------------------------------------------------#



# takes img and box positions, returns boxes drawn img
def draw_boxes(img, boxes, color=(0, 0, 255), thick=6):
    
    # Make a copy of the image
    img_copy = np.copy(img)
    
    # Iterate through the bounding boxes
    for box in boxes:
        
        # Draw a rectangle given box coordinates
        cv2.rectangle(img_copy, box[0], box[1], color, thick)
    
    # Return the image copy with boxes drawn
    return img_copy



#------------------------------------------------------------------------------------------------------#



# takes original image and labeled heatmap tuple, returns labeled box img
def draw_labeled_boxes(image, labels):
    # Copy image
    img = np.copy(image)
    
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Define a bounding box based on min/max x and y
        box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        
        # Draw the box on the image
        cv2.rectangle(img, box[0], box[1], (0,0,255), 6)
        cv2.rectangle(img,(box[0][0],box[0][1]-20),(box[0][0]+100,box[0][1]),(125,125,125),-1)
        cv2.putText(img, 'car {}'.format(car_number),(box[0][0]+5,box[0][1]-2),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0), thickness=2)
    
    # Return the image
    return img


#------------------------------------------------------------------------------------------------------#



def draw_labeled_car_boxes(image, cars):
    img = np.copy(image)
    # Iterate through all detected cars
    for car_number in range(len(cars)):
        bbox = cars[car_number].getBbox()
        if bbox != None:
            cv2.rectangle(img, (bbox[0][0],bbox[0][1]), (bbox[1][0],bbox[1][1]), (0,0,255), 6)
    # Return the image
    return img



#------------------------------------------------------------------------------------------------------#




def raw_and_scaled_windows(sample, imgScales=[1.0, 0.8, 0.65, 0.45], windowOverlap=0.8):
    windows = []
    windows_atScale = []
    
    factor_i = 0
    img_size = sample.shape
    
    
    for scaleFac in imgScales:
        
        inverseFac = 1/scaleFac
        x_scaled = int(img_size[1]*scaleFac) #576
        y_scaled = int(img_size[0]*scaleFac) #324
        img_scaled = cv2.resize(sample, (x_scaled, y_scaled))
        
        
        start_stops = {0:((640,1090), (417,447)), 
                       1:((512,900), (325,370)), 
                       2:((416,832), (265,360)), 
                       3:((288,576), (175,279))}
            
        x_start_stop = start_stops[factor_i][0]
        y_start_stop = start_stops[factor_i][1]
        
        windows_atScale.append(slide_window(img_scaled, x_start_stop=x_start_stop,
                                            y_start_stop=y_start_stop, windowSizeArr=[64], 
                                            xy_overlap=(windowOverlap, windowOverlap)))
        
        # save bounding box in original image space
        for each in windows_atScale[factor_i]:
            windows.append(((int(each[0][0]*inverseFac), int(each[0][1]*inverseFac)),
                            (int(each[1][0]*inverseFac), int(each[1][1]*inverseFac))))
        
        factor_i += 1
        
    return windows_atScale, windows



#------------------------------------------------------------------------------------------------------#




def process_img(img, svc, X_scaler, pca, windows, threshold=1):
    # create heatmap
    heat = np.zeros_like(img[:,:,0].astype(np.float))
    
    # get img patches from frame and extract features
    patches = get_window_imgs(img, windows)
    features = hog_from_list(patches)
        
    # normalize and predict
    X = np.vstack((features)).astype(np.float64)
    scaled_X = X_scaler.transform(X)
    scaled_X = pca.transform(scaled_X)
    pred_bin = svc.predict(scaled_X[:])
    
    # positive positions
    indices = [x for x in range(len(pred_bin)) if pred_bin[x]==1]
    hot_windows = [windows[i] for i in indices]
    
    # draw boxes and heatmap
    # window_img includes all boxes
    # heatmap must shown with cmap='heat'
    # label_img includes generalized boxes only
    img = convert_color(img, 'RGB')
    window_img = draw_boxes(img, hot_windows)
    heat = add_heat(heat, hot_windows)
    heat = apply_threshold(heat, threshold)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    label_img = draw_labeled_boxes(img, labels)
    
    return window_img, label_img, heatmap



#------------------------------------------------------------------------------------------------------#




def process_imgs(imgs, svc, X_scaler, pca, threshold=1, 
                 windowSizes=[96, 128, 145], windowOverlap=0.75):
    window_imgs = []
    label_imgs = []
    heatmaps = []
    
    # sample img for window positions and sample heatmap
    sample = imgs[0]
    
    # declare x/y start stop positions
    x_start_stop=[0, sample.shape[1]]
    y_start_stop=[int(sample.shape[0]/2), sample.shape[0] - 32]
    
    # extract window positions
    windows = slide_window(sample, 
                           x_start_stop=x_start_stop, 
                           y_start_stop=y_start_stop,
                           windowSizeArr=windowSizes, 
                           xy_overlap=(windowOverlap, windowOverlap))
    
    
    # start processing and stack results on seperate lists
    for img in imgs:
        window_img, label_img, heatmap = process_img(img, 
                                                     svc=svc, 
                                                     X_scaler=X_scaler,
                                                     pca=pca, 
                                                     windows=windows, 
                                                     threshold=threshold)
        window_imgs.append(window_img)
        label_imgs.append(label_img)
        heatmaps.append(heatmaps)
    
    return window_imgs, label_imgs, heatmaps



#------------------------------------------------------------------------------------------------------#






