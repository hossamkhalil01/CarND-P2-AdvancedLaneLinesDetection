import matplotlib.pyplot as plt
import matplotlib.image as im
import numpy as np
import cv2
import glob
import pickle
import grad_color_combo as thresh  #Thresholding file

#Undistort the image
def undistortImage (img, cmt,dist):

    #Undistort the image
    undist = cv2.undistort(img,cmt,dist)

    return undist
#Perform Perspective Transform
def BirdsEye_view (img):

    h = img.shape[0]
    w = img.shape[1]



    src = np.array ([
        [202,718],
        [567,460],
        [720,460],
        [1099,718]
    ],np.float32)

    #destination points selection
    dst = np.array ([
        [280,h-1],
        [280,0],
        [950,0],
        [950,h-1]
    ],np.float32)

    #Plot selected points (if needed)
    # plt.imshow(img,cmap='gray')
    # plt.plot(250,670,'.')
    # plt.plot(615,435,'.')
    # plt.plot(661 , 435,'.')
    # plt.plot(1041 , 670,'.')
    # plt.title('Selected Points')
    # plt.show()

    #Perspective transform
    M = cv2.getPerspectiveTransform(src,dst)
    warped = cv2.warpPerspective(img,M,(w,h))

    return warped
#Obtain the color and gradient thresholded binary image
def thresh_combo (img):
    #Combined Gradient
    grad = thresh.grad_combined(img)

    #HLS S channel threshold
    hls_select = thresh.hlsThresh(img, (170,200))


    #Combined Color and Gradient
    combined = np.zeros_like(grad)
    combined [(grad == 1) | (hls_select ==1)] = 1

    return combined

#Find the lanes pixels  using sliding window
def find_lane_pixels (binaryImg):
    #Find the nearst part where lanes are almost vertical
    bottom_half = binaryImg[int(binaryImg.shape[0]/2):,:]

    #Compute histogram for the y axis (height)
    histogram = np.sum(bottom_half,axis = 0)

    # Visualize histogram (if needed)
    # plt.plot(histo)
    # plt.show()

    #Create image to visualize the binary_output
    color_img = np.dstack((binaryImg,binaryImg,binaryImg))

    #Split the histogram
    midpoint = np.int(histogram.shape[0]//2)
    #left peak index
    leftx_base = np.argmax(histogram[:midpoint])
    #right peak index
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #Window Parameters
    N =9                #Number of windows per lane in a frame
    margin = 100        #Half window width value
    pix_mini = 50     #threshold for window recentering

    #Compute window height
    window_height = np.int(binaryImg.shape[0]//N)

    #relevent non zero pixels indices in the frame
    pix_ind = binaryImg.nonzero()
    pix_x = np.array(pix_ind[1])
    pix_y = np.array(pix_ind[0])

    #Window position update for each step
    current_leftx = leftx_base
    current_rightx = rightx_base

    #Lists to hold each lane's pixels indices
    left_lane_ind = []
    right_lane_ind = []

    # Iteration over the N number of windows
    for window in range(1,N+1): #from 1 to number of windows
        #Construct window dimensions
        #Y-axis (common for both lanes)
        y_min = binaryImg.shape[0] - window*window_height
        y_max = y_min + window_height

        #X-axis
        leftx_min = current_leftx - margin
        leftx_max = current_leftx + margin

        rightx_min = current_rightx - margin
        rightx_max = current_rightx + margin

        #Draw the window
        #Left window
        cv2.rectangle(color_img,(leftx_min,y_min),(leftx_max,y_max),(0,255,0), 2)
        #Right window
        cv2.rectangle(color_img,(rightx_min,y_min),(rightx_max,y_max),(0,255,0), 2)

        #Extract non zero pixels indecies in-side each window
        left_ind = ((pix_y >= y_min) & (pix_y < y_max) &
                   (pix_x >= leftx_min) &  (pix_x < leftx_max)).nonzero()[0]

        right_ind = ((pix_y >= y_min) & (pix_y < y_max) &
                    (pix_x >= rightx_min) &  (pix_x < rightx_max)).nonzero()[0]


        #Append the new values in left lane , right lane Lists
        left_lane_ind.append (left_ind)
        right_lane_ind.append(right_ind)

        #Check for window recentering
        if len(left_ind) >  pix_mini:
            #Recenter the left window at the mean position of the pixels
            current_leftx = np.int(np.mean(pix_x [left_ind]))

        if len(right_ind) >  pix_mini:
            #Recenter the right window at the mean position of the pixels
            current_rightx = np.int(np.mean(pix_x [right_ind]))

    # Concatenate the arrays of indices
    left_lane_ind = np.concatenate(left_lane_ind)
    right_lane_ind = np.concatenate(right_lane_ind)


    # Left and Right lane pixels positions
    leftx  = pix_x [left_lane_ind]
    lefty  = pix_y [left_lane_ind]
    rightx = pix_x [right_lane_ind]
    righty = pix_y [right_lane_ind]

    return leftx, lefty, rightx, righty, color_img
#Fit the lanes in a polynomial
def lane_fitting (binaryImg):

    # Find lane pixels
    leftx, lefty, rightx, righty, img = find_lane_pixels(binaryImg)

    #Fit 2nd degree polynomial
    left_fit  = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)

    #Plot the fitted polynomials
    #Y-axis
    y = np.linspace(0, binaryImg.shape[0]-1, binaryImg.shape[0])

    # X = A y**2 + By + C
    #X-axis for left and right lanes
    left_polyx  = left_fit[0]*(y**2) + left_fit[1]*y + left_fit[2]
    right_polyx = right_fit[0]*(y**2) + right_fit[1]*y + right_fit[2]

    #Visualization
    #Visualize the pixels inside the window
    img [lefty , leftx] = [255,0,0]
    img [righty,rightx] = [0,0,255]

    return img, left_polyx ,right_polyx , y



#################
#### Load Data###
#################

#Load test image
testimg = im.imread('test_images/test1.jpg')

plt.figure(1)
plt.imshow(testimg)
plt.title('Original Image')

#Load the camera calibration variables
dist_pickle = pickle.load( open( "camera_cal/calib_mtx.p", "rb" ) )

cmt = dist_pickle["cmt"]
dist = dist_pickle["dist"]


####################
### 1. undistort ###
####################

undist = undistortImage(testimg, cmt, dist)

plt.figure(2)
plt.imshow(undist)
plt.title('Undistorted Image')


###############################################
### 2. Gradient and Color Thresholding ########
###############################################
binary_img = thresh_combo(testimg)

plt.figure(3)
plt.imshow(binary_img,cmap='gray')
plt.title('Gradient and Color Thresholded Binary Image')


######################################
#### 3. Bird's-Eye View Transform ####
######################################
warped_img = BirdsEye_view (binary_img)

plt.figure(4)
plt.imshow(warped_img,cmap='gray')
plt.title('Warped Birds-Eye View Image')



###############################
##### 4. Lane Fitting #########
###############################
lanes_fit , left , right , y= lane_fitting(warped_img)

#Visualize the fitted ploynomials
plt.figure(5)
plt.imshow(lanes_fit,cmap='gray')
plt.title('Fitted Lane Lines')

plt.plot(left, y, color='yellow')
plt.plot(right,y, color='yellow')


plt.show()
#### 4.1. Find Lane Pixels

##### 4.2. Sliding Window
