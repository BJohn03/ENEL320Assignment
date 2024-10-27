import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal


def load_image( infilename ) :
    img = Image.open( infilename ).convert("L")
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def wiener_filter(image_dft, h, K=10):
    H = np.fft.fft2(h)
    W = np.conj(H) / (np.abs(H) **2) + K
    image_dft = image_dft * W
    return image_dft

def gauss(image, rho):
    arr = np.ones(image.shape)
    result = np.ndarray(image.shape)
    
    i = 0
    for x_arr in arr:
        j = 0
        for x in x_arr:
            result[i][j] = gauss_kernel(i - image.shape[0] / 2, j - image.shape[1] / 2 , rho)
            print(i , j)
            j += 1
        i += 1
    
    return result
    
def gauss_kernel(x, y, rho):
    front = 1/ (2*np.pi*(rho**2))
    back = np.exp(-((2*x**2) + (y**2))/(2*rho**2))
    return front * back

def horizontal(image):
    arr = np.ones(image.shape)
    result = np.ndarray(image.shape)
    
    # result[0][0] = 1
    # result[1][1] = 1
    # result[2][2] = 1
    # result[3][3] = 1
    # result[4][4] = 1
    # result[5][5] = 1
    # result[6][6] = 1
    # result[7][7] = 1
    
    # result[image.shape[0] // 2][image.shape[1] // 2] = 1
    # result[image.shape[0] // 2 - 1][image.shape[1] // 2 + 1] = .1
    # result[image.shape[0] // 2 - 2][image.shape[1] // 2 + 2] = .1
    # result[image.shape[0] // 2 - 3][image.shape[1] // 2 + 3] = .1
    # result[image.shape[0] // 2 - 4][image.shape[1] // 2 + 4] = .1
    # result[image.shape[0] // 2 - 5][image.shape[1] // 2 + 5] = .1
    # result[image.shape[0] // 2 - 6][image.shape[1] // 2 + 6] = .1
    # result[image.shape[0] // 2 - 7][image.shape[1] // 2 + 7] = .1
    # result[image.shape[0] // 2 - 8][image.shape[1] // 2 + 8] = .1
    # result[image.shape[0] // 2 - 9][image.shape[1] // 2 + 9] = .1
    # result[image.shape[0] // 2 + 1][image.shape[1] // 2 - 1] = 1
    # result[image.shape[0] // 2 + 2][image.shape[1] // 2 - 2] = 1
    # result[image.shape[0] // 2 + 3][image.shape[1] // 2 - 3] = 1
    # result[image.shape[0] // 2 + 4][image.shape[1] // 2 - 4] = 1
    # result[image.shape[0] // 2 + 5][image.shape[1] // 2 - 5] = 1
    # result[image.shape[0] // 2 + 6][image.shape[1] // 2 - 6] = 1
    # result[image.shape[0] // 2 + 7][image.shape[1] // 2 - 7] = 1
    # result[image.shape[0] // 2 + 8][image.shape[1] // 2 - 8] = 1
    # result[image.shape[0] // 2 + 9][image.shape[1] // 2 - 9] = 1
    # result[image.shape[0] // 2 + 1 ][image.shape[1] // 2 - 10 : image.shape[1] // 2 + 10] = 0.1
    # result[image.shape[0] // 2 - 1 ][image.shape[1] // 2 - 10 : image.shape[1] // 2 + 10] = 0.1

    return result


def main():
    image = load_image("./blurred_car.png")
    
    image_dft = np.fft.fft2(image) 
    h = gauss(image, 1.5)

    plt.imshow(h)
    plt.show()
    
    weiner = wiener_filter(image_dft, h)
    inv_dft = np.fft.ifft2(weiner)
    
    plt.imshow(abs(inv_dft))
    plt.show()
    
if __name__ == "__main__":
    main()
    
    
