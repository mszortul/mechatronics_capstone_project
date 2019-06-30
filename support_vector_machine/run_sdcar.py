from aux_functions import *

heatmap_arr = []
heatmap_filterSize = 8
cars_ar = []
frame_i = 0

sample = cv2.imread('test_images/test1.jpg')

svc = joblib.load('pickles/svm_blurred.pkl')
X_scaler = joblib.load('pickles/svm_scaler_blurred.pkl')
pca = joblib.load('pickles/svm_pca_blurred.pkl')

#svc = joblib.load('svm_no_pca_g.pkl')
#X_scaler = joblib.load('svm_scaler_no_pca_g.pkl')

windows_atScale, windows = raw_and_scaled_windows(sample)

th1 = 3
th2 = 27

def process_frame(img, svc=svc, X_scaler=X_scaler, pca=pca, imgScales=[1.0, 0.8, 0.65, 0.45], 
                  windowOverlap=0.8, windows_atScale=windows_atScale, windows=windows, th1=th1, th2=th2):
    
    # sliding windows creation
    global heatmap_arr
    
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    img_size = img.shape
    imgs = []
    imgCvt = convert_color(img, 'GRAY')
    
    factor_i = 0
    for scaleFac in imgScales:
        
        x_scaled = int(img_size[1]*scaleFac) #576
        y_scaled = int(img_size[0]*scaleFac) #324
        img_scaled = cv2.resize(imgCvt, (x_scaled, y_scaled))
        
        imgs.extend(get_window_imgs(img_scaled, windows_atScale[factor_i], resize=True))
        factor_i += 1
        
    features = hog_from_list_gray(imgs, gaussian_blur=True)#-----------------------------------# expensive comp #1
    X = np.vstack((features)).astype(np.float64)
    scaled_X = X_scaler.transform(X)
    scaled_X = pca.transform(scaled_X) #-------------------------------------------------------# expensive comp #2
    
    pred_bin = svc.predict(scaled_X)
    
    
    
    ind = [x for x in range(len(pred_bin)) if pred_bin[x]==1]
    hot_windows = [windows[i] for i in ind]
    
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)
    
    # Apply threshold to help remove false positives for current frame
    heat = apply_threshold(heat, th1)
    heatmap_arr.append(heat)
    if len(heatmap_arr) > heatmap_filterSize:
        heatmap_arr = heatmap_arr[1:]
    heat_combined = np.zeros_like(img[:,:,0]).astype(np.float)
    
    for i in range(len(heatmap_arr)):
        heat_combined = heat_combined + heatmap_arr[i]
    heat_combined = apply_threshold(heat_combined, th2)
    
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat_combined, 0, 255)
    #heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    
    return heatmap, labels, hot_windows




def process_vidFrame(img, outputDebug=False, boxOnlyOutput=False):
    
    global frame_i
    global cars_ar
    frame_i += 1
    
    heatmap, labels, hot_windows = process_frame(img)
    
    
    for car_number in range(1, max(len(cars_ar), labels[1])+1):
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        if len(nonzerox):
            # cut off tiny bounding boxes
            if ((max(nonzerox) - min(nonzerox)) / (max(nonzeroy) - min(nonzeroy))) > 0.65:
                # Define a bounding box based on min/max x and y
                bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            else:
                bbox = None
        else:
            bbox = None
        if len(cars_ar) < car_number:
            cars_ar.append(Car())
        cars_ar[car_number-1].updatePos(bbox)
    
    
    #label_img = vehicleUtil.draw_labeled_bboxes(np.copy(img), labels)
    label_img = draw_labeled_car_boxes(img, cars_ar)
    
    if outputDebug:
        imgSize = (720, 1280 , 3)
        out_img = np.zeros(imgSize, dtype=np.uint8)

        smallFinal = cv2.resize(label_img, (0,0), fx=0.5, fy=0.5)
        smallFinalSize = (smallFinal.shape[1], smallFinal.shape[0])
        out_img[0:smallFinalSize[1], 0:smallFinalSize[0]] = smallFinal

        heatmap = heatmap*(255/8)
        heatmap = np.clip(heatmap, 0, 255)
        heatmap = np.dstack((heatmap, heatmap, heatmap))
        smallHeat = cv2.resize(heatmap, (0,0), fx=0.5, fy=0.5)
        smallHeatSize = (smallHeat.shape[1], smallHeat.shape[0])
        out_img[0:smallHeatSize[1], smallFinalSize[0]:smallFinalSize[0]+smallHeatSize[0]] = smallHeat

        window_img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)
        rawWindows = cv2.resize(window_img, (0,0), fx=0.5, fy=0.5)
        rawWindowsSize = (rawWindows.shape[1], rawWindows.shape[0])
        out_img[smallFinalSize[1]:smallFinalSize[1]+rawWindowsSize[1], smallFinalSize[0]:smallFinalSize[0]+rawWindowsSize[0]] = rawWindows
    else:
        img = convert_color(label_img, 'RGB')
        window_img = draw_boxes(img, hot_windows)
        out_img = label_img
    
    if boxOnlyOutput:
        return cars_ar
    return out_img



file = 'project_video.mp4'
clip = VideoFileClip(file)

proc_clip = clip.fl_image(process_vidFrame)
#proc_output = '{}_proc_lab_'+str(th1)+str(th2)+'.mp4'.format(file.split('.')[0])
proc_output = 'project_video_proc_allblur5gauss_'+str(th1)+str(th2)+'.mp4'
proc_clip.write_videofile(proc_output, audio=False)