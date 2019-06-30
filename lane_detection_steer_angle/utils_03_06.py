from utils_01_06 import *


def solver(edge_img, YM=YM):
    '''
        Get weight matrix of parabolic function from single edge image

        Parameters:

            edge_img : np.array, single edge image

            YM : scalar, near far field threshold

        Return:

            C : np.array, [a, b, d]T weight matrix, shape of (3, 1)

    '''
    
    far_field = np.zeros_like(edge_img)
    near_field = np.zeros_like(edge_img)

    far_field[:YM, :] = edge_img[:YM, :]
    near_field[YM:, :] = edge_img[YM:, :]
    
    far_non = np.nonzero(far_field)
    near_non = np.nonzero(near_field)
    tfar_non = np.transpose(far_non)
    tnear_non = np.transpose(near_non)
    
    n = tfar_non.shape[0]
    m = tnear_non.shape[0]
    
    mag_n = far_field[far_non]
    mag_f = near_field[near_non]
    mag_nf = np.hstack((mag_n, mag_f))

    A = np.zeros((n+m, 3))
    W = np.diag(mag_nf)  
    C = np.zeros((3, 1))
    B = np.zeros((n+m, 1))
    
    A[:, 0] = 1
    A[:m, 1] = tnear_non[:, 0]#.reshape((m, 1))
    B[:m, 0] = tnear_non[:, 1]#.reshape((m, 1))
    B[m:, 0] = tfar_non[:, 1]#.reshape((n, 1))
    yf = tfar_non[:, 0]#.reshape((n, 1))
    yf2 = np.power(yf, 2)
    
    A1 = (yf2 + math.pow(YM, 2)) / (2 * YM)
    A2 = (-1) * np.power((yf - YM), 2) / (2 * YM)
    
    A[m:, 1] = A1
    A[m:, 2] = A2
    f_side = np.matmul(np.matmul(np.transpose(A), W), A)
    s_side = np.matmul(np.matmul(np.transpose(A), W), B)
    f_side_inv = inv(f_side)
    C = np.matmul(f_side_inv, s_side)
    
    return C




def quad_linear(C, y, YM=YM):
    '''
        Puts weights into function

        Parameter:

            C : np.array, weight matrix

            y : scalar, y value

            YM : scalar, near far field threshold

        Return:

            x_val: scalar, x value

    '''
    a = C[0, 0]
    b = C[1, 0]
    d = C[2, 0]
    
    if (y > YM):
        x_val = a + b*y
        return x_val
    else:
        x_val = ((a + (YM/2) * (b - d))) + d * y + (math.pow(y, 2) / (2 * YM)) * (b-d)
        return x_val