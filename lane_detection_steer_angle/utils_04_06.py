from utils_03_06 import *
from hough import *



def parabola_image(edge_image, C, lw=3):
    '''
        Get parabola image with function weights

        Parameters:

            edge_image : np.array, single edge image

            C : np.array, weight matrix

            lw : scalar, width of parabola

        Return:

            plot_image : np.array, parabola drawn on black ground, shape of edge_image


    '''
    
    y = np.linspace(0, 720, 1440)
    x = np.zeros_like(y)

    for i in range(y.shape[0]):
        x[i] = quad_linear(C, y[i])
        
    blank = np.zeros_like(edge_image).astype(np.uint8)
    
    fig, ax = plt.subplots()
    ax.imshow(blank, plt.get_cmap('gray'))
    ax.plot(x, y, 'w-', linewidth=lw)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    d = data[49:238, 53:390] # crop plot
    plot_image = cv2.resize(d, (1280, 720)) # resize to original dims
    plot_image = rgb2gray(plot_image)
    
    plt.close(fig)
    
    return plot_image, x


def get_edges(aeim):
    '''
        Extract edges from aeim
        
        Parameters:
        
            aeim : tuple, Returned from get_new_angles_edges()
            
        Return:
            
            edges : list, list of edge images
    
    '''
    
    edges = []
    
    for i in range(len(aeim)):
        edges.append(aeim[i][1])
        
    return edges


def get_parabolas(edges, lw=3, hist=True):
    '''
        Get parabolas from edge images

        Parameters:

            edges : list, list of edge images

            lw : scalar, width of parabola

        Return:

            parabolas : list, parabola images list

    '''
    
    parabolas = []
    mid_y = np.linspace(0, 720, 1440)
    xs = []
    
    for edge in edges:
        if hist:
            C = solver(edge)
            parabola, x = parabola_image(edge, C, lw=lw)
        else:
            edge_im = np.zeros((720, 1280))
            edge_im[470:650, :] = edge
            C = solver(edge_im)
            parabola, x = parabola_image(edge_im, C, lw=lw)
        
        xs.append(x)
        parabolas.append(parabola)
        
    mid_x = (xs[0] + xs[1]) / 2
    
    return parabolas, mid_y, mid_x



'''
    
    IPM FUNCTIONS
    
    Little blurring and a little bit of cheese, bon apetite!

'''


def build_ipm_table(srcw, srch, dstw, dsth, vptx, vpty, maptable):
    
    CAMERA_POS_Y = 0  # d (cm)
    CAMERA_POS_X = 0  # l (cm)
    CAMERA_POS_Z = 60 # h (cm)
    FOV_H = 80.0   # (degree)
    FOV_V = 50.0   # (degree)
    DEG2RAD = 0.01745329252
    
    alpha_h = 0.5 * FOV_H * DEG2RAD
    alpha_v = 0.5 * FOV_V * DEG2RAD
    gamma = -float((vptx - int(srcw / 2)) * (alpha_h / int(srcw / 2)))
    theta = -float((vpty - int(srch / 2)) * (alpha_v / int(srch / 2)))
    
    front_map_start_position = int(dsth / 2)
    front_map_end_position = front_map_start_position + dsth
    side_map_mid_position = int(dstw / 2)
    
    front_map_scale_factor = 4
    side_map_scale_factor = 2
    
    for y in range(dstw):
        for x in range(front_map_start_position, front_map_end_position):
            idx = y * dsth + (x - front_map_start_position)
            deltax = front_map_scale_factor * (front_map_end_position - x - CAMERA_POS_X)
            deltay = side_map_scale_factor * (y - side_map_mid_position - CAMERA_POS_Y)
            
            if deltay is 0:
                maptable[idx] = maptable[idx - dsth]
            else:
                u = int((atan(CAMERA_POS_Z * sin(atan(float(deltay) / deltax)) / deltay) - (theta - alpha_v)) / (2 * alpha_v / srch))
                v = int((atan(float(deltay) / deltax) - (gamma - alpha_h)) / (2 * alpha_h / srcw))
                
                if ((u >= 0) and (u < srch) and (v >= 0) and (v < srcw)):
                    maptable[idx] = srcw * u + v
                else:
                    maptable[idx] = -1
                    
                    
def inverse_perspective_mapping(dstw, dsth, src, maptable, dst):
    idx = 0
    for j in range(dsth):
        for i in range(dstw):
            if maptable[idx] is not -1:
                dst[i * dsth + j] = src[maptable[idx]]
            else:
                dst[i * dsth + j] = 0
            idx += 1
            
            
def ipm(im, dy=0):
    '''
        The main ipm function, upper two are utility
        
        Parameters:
        
            im : np.array, original RGB image with little recommended blur
            
            dy : scalar, vanishing point adjustment value, directly added onto (height/2)
            
        Return:
        
            imremapped : np.array, grayscale IPM image, shape of (200, 200)
            
            imresize : np.array, RGB resized image, shape of (360, 720)
    
    '''
    
    SRC_RESIZED_WIDTH = 720
    SRC_RESIZED_HEIGHT = 360
    DST_REMAPPED_WIDTH = 200
    DST_REMAPPED_HEIGHT = 200

    vanishing_point_x = SRC_RESIZED_WIDTH / 2
    vanishing_point_y = SRC_RESIZED_HEIGHT / 2 + dy

    ipm_table = np.zeros((DST_REMAPPED_WIDTH * DST_REMAPPED_HEIGHT), dtype=np.int32)
    build_ipm_table(SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT,
                    DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT,
                    vanishing_point_x, vanishing_point_y, ipm_table)

    imremapped = np.zeros((DST_REMAPPED_HEIGHT, DST_REMAPPED_WIDTH), dtype=np.uint8)
    imresize = cv2.resize(im, (SRC_RESIZED_WIDTH, SRC_RESIZED_HEIGHT))
    grayresize = rgb2gray(imresize)

    inverse_perspective_mapping(DST_REMAPPED_WIDTH, DST_REMAPPED_HEIGHT, np.ravel(grayresize), ipm_table, np.ravel(imremapped))
    
    return imremapped, imresize




def crop_ipm(parabolas, ymin=450, ymax=650, dy=0):
    '''
        Crop parabolas, draw them and apply ipm

        Parameters:

            parabolas : list of parabola images

            ymin : lower y crop value

            ymax : upper y crop value

        Return:

            rm : ipm image

            rs : resized original image (RGB IMAGE ALERT!!)

    '''
    
    comp = np.zeros_like(parabolas[0])
    for ind in range(len(parabolas)):
        comp[ymin:ymax, :] += parabolas[ind][ymin:ymax, :]
    
    comp *= (255.0/np.max(comp))
    comp = comp.astype(np.uint8)
    
    comp_rgb = color.gray2rgb(comp)
    
    comp_rgb = cv2.GaussianBlur(comp_rgb, (7,7), 3)
    rm, rs = ipm(comp_rgb, dy)
    
    return rm, rs, comp



def turn(img):
    rm, rs = ipm(img, 20)
    cropped = rm[:150, 50:150]
    canny = cv2.Canny(cropped, 50, 100)
    acc, diag, thetas = houghcraft(canny)
    rho_theta, theta_edf = find_peaks(acc, diag, thetas)
    
    ang = (rho_theta[0][1] + rho_theta[1][1])
    
    return ang


def ang_vid(start, stop, fname):
    outpath = 'video_out/' + fname + '.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outpath, fourcc, 25.0, (1280, 720))
    
    font = ImageFont.truetype(font="font/FiraMono-Medium.otf",size=25)
    
    for i in range(start, stop):
        im = mpimg.imread('proframe_imgs/' + str(i) + '.jpg')
        #im = np.stack([im[:,:,2], im[:,:,1], im[:,:,0]], axis=2)
        ang = turn(im)
        
        pimg = Image.fromarray(im, 'RGB')
        draw = ImageDraw.Draw(pimg)
        draw.text((0,0), str(i) + '. frame: ' + str(ang), (255,0,0), font=font)
        draw = ImageDraw.Draw(pimg)
        
        im_save = np.asarray(pimg)
        
        out.write(im_save)
        