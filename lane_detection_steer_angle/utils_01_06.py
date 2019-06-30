from utils_31_05 import *


'''

aeim = get_new_angles_edges(img_orig, lbrois)
gray_lines, new_lbrois = lbroi_draw_on_img(gray, aeim)

'''

def lbroi_indexes(lbroi):
    '''
        Get pixel indexes of lbroi
        
        Parameters:
        
            lbroi : np.array, lbroi image
            
        Return:
        
            indexes : np.array, index array, shape of (number of pixels, 2)
    
    '''
    
    indexes = np.transpose(np.nonzero(lbroi))
    return indexes

def lbroi_tan_mag(gx, gy, indexes):
    '''
        Apply tan_mag() in lbroi
    
    '''
    
    h, w = gx.shape
    
    tans = np.zeros((h,w))
    mags = np.zeros((h,w))
    for i in range(indexes.shape[0]):
        y = int(indexes[i, 0])
        x = int(indexes[i, 1])
        tan = rad2deg(math.atan(-gx[y,x] / gy[y,x]))
        tans[y,x] = tan
        mag = math.sqrt(gy[y,x]**2 + gx[y,x]**2)
        mags[y,x] = mag

    return tans, mags


def lbroi_plot_vars(tans, mags, indexes):
    '''
        Apply plot_vars() in lbroi
    
    '''
    
    #print(tans.shape)
    tan_arr = np.zeros_like(tans)
    mag_arr = np.zeros_like(mags)
    
    mags = th_mag_arr(mags, indexes)
    
    #print(tan_arr.shape)
    h, w = tans.shape
    hist = {}
    xs = np.arange(-90, 91, 2)
    for x in xs:
        hist[x] = 0
    
    for i in range(indexes.shape[0]):
        #print(indexes[i].shape)
        #print(indexes.shape)
        y = int(indexes[i, 0])
        x = int(indexes[i, 1])
        #print(tans[y].shape)
        #print(tans[x].shape)
        #print(tans.shape)
        if (np.isnan(tans[y,x])):
            continue
        else:
            tan_arr[y,x] = tans[y,x]
            mag_arr[y,x] = mags[y,x]
            if (int(tans[y,x]) % 2 is 0):
                hist[int(tans[y,x])] += mags[y,x]
            else:
                hist[int(tans[y,x]) - 1] += mags[y,x]
    
    return hist, tan_arr, mag_arr

def th_mag_arr(mag_arr, indexes, th=0.5):
    '''
        Apply thresholding within lbroi
        
        Parameters:
        
            mag_arr : np.array, return of lbroi_plot_vars()
            
            indexes : np.array, indexes of lbroi pixels
            
            th : scalar, threshold ratio between 0 to 1
            
        Return:
        
            th_mag_arr : np.array, threshold mag_arr from input
    
    '''
    
    mag_sum = np.sum(mag_arr)
    px_num = indexes.shape[0]
    mean = mag_sum/px_num
    
    th_mag_arr = np.multiply(mag_arr, mag_arr>mean)
    
    return th_mag_arr

def lbroi_one_edge_edf(img, lbroi, sigma=7, ret=True, plott=False):
    '''
        Almost same as edf_comp(), except appyls only within one lbroi
    
    '''
    
    indexes = lbroi_indexes(lbroi)
    gray_blur = pre(img)
    gx, gy = sobel(gray_blur)
    tans, mags = lbroi_tan_mag(gx, gy, indexes)
    hist, tan_arr, mag_arr = lbroi_plot_vars(tans, mags, indexes)
    x, y = edf_smooth(hist, sigma=sigma)
    #print(x.shape)
    #print(y.shape)
    if plott:
        plt.plot(x, y)
        plt.show()
    if ret:
        return y, tan_arr, mag_arr, indexes
    
    
def lbroi_get_maxima(y, ex_ang1, lr=4):
    '''
        Almost same as get_maxima(), except it finds only the global maxima
    
    '''
    x = np.linspace(-90, 90, 361)
    
    ex_ang = sum(ex_ang1) / len(ex_ang1)
    
    mmax = np.where(x == int(ex_ang))
    ind = mmax[0][0]
    #print(ind)
    
    if ind>=lr:
        left = ind-lr
    else:
        left = 0
    if ind<=360-lr:
        right = ind+lr
    else:
        right = x.shape[0]
        
    frac = y[left:right]
    
    m = np.argmax(frac)
    
    global_max = x[left+m]
    
    #maxm = argrelextrema(y, np.greater)
    #global_max = x[np.argmax(y)]
    return global_max


def lbroi_single_edge_image(tan_arr, mag_arr, alpha, indexes):
    '''
        Almost same as single_edge_image()
    
    '''
    
    h, w = tan_arr.shape
    single_edge_image = np.zeros_like(mag_arr)
    for i in range(indexes.shape[0]):
        y = indexes[i, 0]
        x = indexes[i, 1]
        if abs(tan_arr[y,x] - alpha) < 2.0:
            single_edge_image[y,x] = mag_arr[y,x]
                
    return single_edge_image

def get_new_angles_edges(img, lbrois, ang_buf):
    '''
        Get angles, edge images, indexes and magnitude arrays
        
        Parameters:
            
            img : np.array, original RGB image
            
            lbrois : list, detected lbrois from last frame
            
        Return:
        
            aeim : list, [edge_angle, edge, indexes, mag_arr] length of #detected_lines
                
                edge_angle : scalar, edf angle of detected edge
                
                edge : np.array, single edge image
                
                indexes : np.array, lbroi pixels
                
                mag_arr : np.array, thresholded magnitude arrays
    
    '''
    
    aeim = []
    for i in range(len(lbrois)):
        lb = lbrois[i]
        ex_ang = ang_buf[i]
        y, tan_arr, mag_arr, indexes = lbroi_one_edge_edf(img, lb)
        edge_angle = lbroi_get_maxima(y, ex_ang)
        edge = lbroi_single_edge_image(tan_arr, mag_arr, edge_angle, indexes)
        aeim.append([edge_angle, edge, indexes, mag_arr])
        
    aeim.sort(key=itemgetter(0))
    
    return aeim



def hough_angles(aeim1):
    '''
        Convert from edf angles to hough angles
        
        Parameters:
        
            aeim1 : list, return of get_new_angles_edges()
            
        Return:
        
            aeim : list, angles are modified for hough transform
    
    '''
    
    aeim = copy.deepcopy(aeim1)
    for i in range(len(aeim)):
        a = aeim[i][0]
        if a < 0:
            aeim[i][0] = -90 - a
        else:
            aeim[i][0] = 90 - a
    
    return aeim


def lbroi_weighted_hough(edge_img, theta, mag_arr, indexes):
    '''
        Almost same as weighted_hough(), except it only applies within lbroi
    
    '''
    
    t = deg2rad(theta)
    h, w = edge_img.shape
    diag = int(math.ceil(math.sqrt(h**2 + w**2)))
    acc = np.zeros(2*diag+1)
    for i in range(indexes.shape[0]):
        y = int(indexes[i, 0])
        x = int(indexes[i, 1])
        if edge_img[y,x] > 0:
            rho = int((y * math.sin(t) + x * math.cos(t)) + diag)
            acc[rho] += mag_arr[y,x]
                
    return acc, diag, theta


def lbroi_line(rho, alpha, h=720, w=1280, thicc=5):
    '''
        Slightly differs from line(), no biggie tho lol!
    
    '''
    
    line_im = np.zeros((h, w)).astype(np.uint8)
    p = []
    for x in range(w):
        y = int( -x / math.tan(alpha * (math.pi/180)) + (rho / (math.sin(alpha*(math.pi/180)))))
        if (y < h and y >= 0):
            p.append((x, y))
    
    p.sort(key=itemgetter(1))
    
    #print('orig', p[0], p[last])
    
    ps = (p[0], p[-1])
    #ps = calc_p((p[-1], p[0]))
    
    #print('new', ps)
    cv2.line(line_im, ps[0], ps[1], (255,255,255), thickness=thicc)
    return line_im[450:650,:]


def lbroi_draw_on_img(gray, aeim1, y1=470, y2=650, thicc=20):
    '''
        Almost same as draw_on_img()
        
        Parameters:
        
            gray : np.array, grayscale image
            
            aeim1 : list, return of get_new_angles_edges()
            
            y1 : scalar, same as draw_on_img()
            
            y2 : scalar, same as draw_on_img()
            
            thicc : scalar, same as draw_on_img()
            
        Return:
            
            gray_lines : np.array, same as draw_on_img()
            
            lbrois : list, same as draw_on_img()
    
    '''
    angs = []
    for i in range(len(aeim1)):
        angs.append(aeim1[i][0])
    gray_lines = np.copy(gray)
    #h, w = (470,1280)
    aeim = hough_angles(aeim1)
    lbrois = []
    
    for i in range(len(aeim)):
        
        
        lbroi = np.zeros_like(gray)
        mag_arr = aeim[i][3]
        indexes = aeim[i][2]
        edge_im = aeim[i][1]
        angle = aeim[i][0]
        
        #gshow(edge_im)
        #print(i, 'angle:', angle)
        #gshow(mag_arr)
        #print(i, 'mag_arr.shape:', mag_arr.shape)
        acc, diag, theta = lbroi_weighted_hough(edge_im, angle, mag_arr, indexes)
        
        smooth_acc = gaussian_filter1d(acc, sigma=29)
        x = np.arange(0, 2*diag+1, 1)
        
        global_max = x[np.argmax(smooth_acc)]
        
        rho = global_max - diag
        
        line_img = lbroi_line(rho, angle, thicc=thicc)
        
        coors = np.zeros(4).astype(np.uint32)
        
        coors[0] = y1-EXTENSION
        coors[1] = y2
        coors[2] = 0 
        coors[3] = 1280
        
        #for g in range(4):
        #    coors[g] = int(coors[g])
        #    print(coors[g])
        #for j in range(4):
        #    print(coors[j])
        
        #print(lbroi.shape)
        #print(line_img.shape)
        #print(i, 'line_img: ')
        #gshow(line_img)
        
        lbroi[coors[0]:coors[1], coors[2]:coors[3]] += line_img
        lbrois.append(lbroi)
        gray_lines[coors[0]:coors[1], coors[2]:coors[3]] += line_img
    
    return gray_lines, lbrois, angs


def angbuf(ang_buf, angs):
    for i in range(2):
        ang_buf[i].append(angs[i])
    if (len(ang_buf[0]) > 4):
        ang_buf[0].popleft()
        ang_buf[1].popleft()
    
    return ang_buf