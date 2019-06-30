from imports import *


def houghcraft(img):
    h, w = img.shape
    thetas = np.deg2rad(np.arange(-90, 90, 1))
    diag = int(math.ceil(math.sqrt(h**2 + w**2)))
    #rhos = np.linspace(-diag, diag, diag*2+1)
    
    cos_vals = np.cos(thetas)
    sin_vals = np.sin(thetas)
    lt = thetas.shape[0]
    
    acc = np.zeros((2 * diag + 1, lt))
    
    for y in range(h):
        for x in range(w):
            if (img[y,x] > 0):
                for deg in range(lt):
                    rho = int((y * sin_vals[deg] + x * cos_vals[deg]) + diag) 
                    acc[rho, deg] += 1
                    
    return acc, diag, np.rad2deg(thetas)


def find_peaks(acc, diag, thetas):
    blur = cv2.GaussianBlur(acc, (3,3), 3)
    coor = peak_local_max(img_as_float(blur), min_distance=15, num_peaks=2)
    rho_draw = coor[:,0] - diag
    theta_draw = thetas[coor[:,1]]
    theta_edf = np.zeros_like(theta_draw)
    tlen = theta_draw.shape[0]
    for i in range(tlen):
        curr = theta_draw[i]
        if curr < 0:
            theta_edf[i] = -90 - curr
        else:
            theta_edf[i] = 90 - curr
            
    rho_theta = np.hstack((rho_draw.reshape((tlen, 1)), theta_draw.reshape((tlen, 1))))
    
    return rho_theta, theta_edf


def hline(rho, alpha, h, w, thicc=5, ext=True):
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
        try:    
            y = int( -x / math.tan(alpha * (math.pi/180)) + (rho / (math.sin(alpha*(math.pi/180)))))
        except:
            continue
        if (y < h and y >= 0):
            p.append((x, y))
    
    p.sort(key=itemgetter(1))
    
    #print('new', ps)
    cv2.line(line_im, p[0], p[-1], (255,255,255), thickness=thicc)
    return line_im