from edf_u import *

def get_maxima(y):
    x = np.linspace(-90, 90, 360)
    maxm = argrelextrema(y, np.greater)
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


def weighted_hough(edge_img, alpha, mag_arr):
    h, w = edge_img.shape
    acc_arr = np.zeros(int(math.sqrt(h**2 + w**2)))
    for y in range(h):
        for x in range(w):
            if edge_img[y,x] is not 0:
                a = alpha * (math.pi/180)
                rho = y * math.sin(a) + x * math.cos(a)
                acc_arr[rho] += mag_arr[y,x]
    
    return acc_arr

def smooth_line(arr, f_size):
    filt = np.ones(f_size) / f_size
    for i in range(int(f_size/2), arr.shape[0] - int(f_size/2)):
        new_val = sum(np.multiply(filt, arr[i-int(f_size/2):i+int(f_size/2)+1]))
        arr[i] = new_val
    
    return arr

def line(rho, alpha, h, w):
    line_im = np.zeros((h, w)).astype(np.uint8)
    for x in range(w):
        y = int( -x / math.tan(alpha * (math.pi/180)) + (rho / (math.sin(alpha*(math.pi/180)))))
        if (y < h and y >= 0):
            line_im[y,x] = 255
    
    return line_im

