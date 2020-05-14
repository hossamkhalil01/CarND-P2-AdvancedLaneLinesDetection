import numpy as np
import cv2
import matplotlib.image as im

def grad_xy (gray, orient = 1, thresh = ( 0 , 255)  , ksize = 3 ):


    if orient ==1 :
        dx = 1
        dy = 0
    else:
        dx = 0
        dy = 1

    #sobel
    sobel = np.absolute(cv2.Sobel(gray,cv2.CV_64F,dx,dy,ksize= ksize))

    #Rescale
    sobel_scale = np.uint8((255*sobel)/np.max(sobel))

    #Thresholding
    binary_output = np.zeros_like(gray)
    binary_output [(sobel_scale >= thresh[0]) & (sobel_scale <= thresh[1])] = 1

    return binary_output

def grad_magn (gray, thresh = ( 0 ,  255) , ksize = 3):

    #Sobelx
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0)

    #Sobely
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1)

    #Magintue
    sobelxy = np.sqrt(sobelx**2+sobely**2)

    #Rescale
    sobelmagn = np.uint8((sobelxy*255)/np.max(sobelxy))

    #Binary
    binary_output = np.zeros_like(gray)
    binary_output [ (sobelmagn >= thresh[0]) & (sobelmagn <= thresh[1])] =1

    return binary_output

def grad_dir (gray, thresh = (0 , np.pi/2) , ksize = 11):

    #sobel x
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize)

    #sobel y
    sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize)

    #Gradient Direction
    gradDir = np.arctan2(np.absolute(sobely),np.absolute(sobelx))


    #Binary output
    binary_output = np.zeros_like(gray)
    binary_output [ (gradDir >= thresh[0]) & (gradDir <= thresh[1])] =1

    return binary_output

def grad_combined (img):

    #grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    #Sobelx
    sobelx = grad_xy (gray,1,(30,255),15)

    #Sobely
    sobely = grad_xy (gray,0,(30,255),15)

    #Magnitude
    sobelMagn = grad_magn (gray,(30,255),15)

    #Gradient Direction
    gradDir = grad_dir (gray,(0.7,1.3),15)

    #Combined mask
    combined = np.zeros_like(gray)
    combined [ ((sobelx == 1) & (sobely == 1)) |\
              ((sobelMagn ==1) & (gradDir ==1)) ]  =1

    return combined

def hlsThresh (img , thresh = (0,255)):
    #Convert to hls color space
    hls = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)


    #Select S channel
    s = hls[:,:,2]

    #Create binary output image
    hls_binary = np.zeros_like (s)
    hls_binary [ (s >= thresh[0]) & (s <= thresh[1])] =1

    return hls_binary
