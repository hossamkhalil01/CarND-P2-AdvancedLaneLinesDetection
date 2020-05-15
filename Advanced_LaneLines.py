import matplotlib.pyplot as plt
import matplotlib.image as im
import numpy as np
import cv2
import glob
import pickle
from numpy.linalg import inv
from moviepy.editor import VideoFileClip
import grad_color_combo as thresh  #Thresholding file

# #To update the parameters of each frame 
# class Line():
#     def __init__(self):
#         # was the line detected in the last iteration?
#         self.detected = False
#         # x values of the last n fits of the line
#         self.recent_xfitted = []
#         #average x values of the fitted line over the last n iterations
#         self.bestx = None
#         #polynomial coefficients averaged over the last n iterations
#         self.best_fit = None
#         #polynomial coefficients for the most recent fit
#         self.current_fit = [np.array([False])]
#         #radius of curvature of the line in some units
#         self.radius_of_curvature = None
#         #distance in meters of vehicle center from the line
#         self.line_base_pos = None
#         #difference in fit coefficients between last and new fits
#         self.diffs = np.array([0,0,0], dtype='float')
#         #x values for detected line pixels
#         self.allx = None
#         #y values for detected line pixels
#         self.ally = None
#Undistort the image
def undistortImage (img, cmt,dist):

    #Undistort the image
    undist = cv2.undistort(img,cmt,dist)

    return undist
#Perform Perspective Transform
def BirdsEye_view (img):

    h = img.shape[0]
    w = img.shape[1]

    # src = np.array ([
    #     [202,718],
    #     [567,460],
    #     [720,460],
    #     [1099,718]
    # ],np.float32)

    src = np.array ([
        [200,719],
        [580,460],
        [710,460],
        [1106,719]
    ],np.float32)
    #destination points selection
    dst = np.array ([
        [320,h-1],
        [320,0],
        [960,0],
        [960,h-1]
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

    return M,warped
#Obtain the color and gradient thresholded binary image
def thresh_combo (img):
    #Combined Gradient
    grad = thresh.grad_combined(img)

    #HLS S channel threshold
    hls_select = thresh.hlsThresh(img, (180,240))


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
    margin = 70        #Half window width value
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
#Compute curvature radius and offset from center
def compute_curvatureAndOffset(width,y,left_fit,right_fit):
    #from pixel to meters conversion
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/608 # meters per pixel in x dimension

    #################
    ###1. Offset ####
    #################
    #Get x position for left and right lanes at the max bottom of the image
    xleft  = left_fit[len(left_fit)-1]
    xright = right_fit[len(right_fit)-1]

    #Lane Center
    lane_center = ((xright-xleft)/2) + xleft
    img_center = width/2

    #offset (in pixels)
    offset = img_center - lane_center
    #offset (in meters)
    offset_m = offset*xm_per_pix

    #####################
    #### 2. Curvature ###
    #####################
    #Fit the polynomial in meters
    left_fit_m = np.polyfit(y*ym_per_pix , left_fit*xm_per_pix , 2)
    right_fit_m = np.polyfit(y*ym_per_pix , right_fit*xm_per_pix , 2)

    #Calculate for the largest y value (bottom half)
    y_eval = np.max(y)

    #Compute the radius of the curve at the max y point using curvature  radius eqn.
    #Calculation of the left lane
    left_curvR  = (np.sqrt((1+((2*left_fit_m[0]*y_eval*ym_per_pix)+left_fit_m[1])**2)**3))/(np.absolute(2*left_fit_m[0]))
    #Calculation of the right lane
    right_curvR = (np.sqrt((1+((2*right_fit_m[0]*y_eval*ym_per_pix)+right_fit_m[1])**2)**3))/(np.absolute(2*right_fit_m[0]))

    return left_curvR, right_curvR,offset_m
#Draw the results on the undistort image
def drawResults (undistImg, warpedImg, mtx,poly_y , left_fitx, right_fitx):

    # Color image to draw the lines on
    gray_warp = np.zeros_like(warpedImg).astype(np.uint8)
    color_warp = np.dstack((gray_warp, gray_warp, gray_warp))

    #init points for creating poly
    pts_left  = np.array([np.transpose(np.vstack([left_fitx, poly_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, poly_y])))])
    poly_pts  = np.hstack((pts_left, pts_right))

    #Draw the lane in warped image (in blue)
    cv2.fillPoly(color_warp, np.int_([poly_pts]), (0,0, 255))

    #Warp the image back to the original perspective (using the transform matrix inverse(mtx))
    unwarp = cv2.warpPerspective(color_warp, inv(mtx), (undistImg.shape[1], undistImg.shape[0]))

    #Result
    result = cv2.addWeighted(undistImg, 1, unwarp, 0.3, 0)

    return result
#Image Lane detection pipline
def Lane_detection(img):

    #Load the camera calibration variables
    dist_pickle = pickle.load( open( "camera_cal/calib_mtx.p", "rb" ) )

    cmt = dist_pickle["cmt"]
    dist = dist_pickle["dist"]

    # #Visualize Original Image if needed
    # plt.figure(1)
    # plt.imshow(img)
    # plt.title('Original Image')

    ####################
    ### 1. undistort ###
    ####################

    undist_img = undistortImage(img, cmt, dist)

    # #Visualize and save image if needed
    # plt.figure(2)
    # plt.imshow(undist_img)
    # plt.title('Undistorted Image')
    #
    # #Save Image
    # plt.imsave('output_images/undistorted.jpg', undist_img)


    ###############################################
    ### 2. Gradient and Color Thresholding ########
    ###############################################
    binary_img = thresh_combo(undist_img)

    # #Visualize and save image if needed
    # plt.figure(3)
    # plt.imshow(binary_img,cmap='gray')
    # plt.title('Gradient and Color Thresholded Binary Image')
    #
    # #Save Image
    # plt.imsave('output_images/combo_thresh.jpg', binary_img,cmap='gray')
    #

    ######################################
    #### 3. Bird's-Eye View Transform ####
    ######################################
    mtx,warped_img = BirdsEye_view (binary_img)

    # #Visualize and save image if needed
    # plt.figure(4)
    # plt.imshow(warped_img,cmap='gray')
    # plt.title('Warped Birds-Eye View Image')
    #
    # #Save Image
    # plt.imsave('output_images/BirdEye_view.jpg', warped_img,cmap='gray')

    ################################
    ###### 4. Lane Fitting #########
    ################################
    fit_img , left , right , poly_y = lane_fitting(warped_img)

    #Visualize the fitted ploynomials
    # #Visualize and save image if needed
    # plt.figure(5)
    # plt.imshow(fit_img,cmap='gray')
    # plt.title('Fitted Lane Lines')
    # plt.plot(left, poly_y, color='yellow')
    # plt.plot(right,poly_y, color='yellow')
    #
    # #Save Image
    # plt.savefig('output_images/Fitted_LaneLines.jpg')

    #######################################################
    ###### 6. Compute Curvature and Cetner Offset #########
    #######################################################

    left_curvR , right_curvR , offset = compute_curvatureAndOffset(undist_img.shape[1],poly_y,left,right)
    #Mean of left and right radius
    mean_curvR = (left_curvR+right_curvR)/2


    ############################################################
    ###### 7. Draw lines on origianl undistorted Image #########
    ############################################################

    final_img =  drawResults (undist_img, warped_img, mtx, poly_y, left, right)

    #Print the output values on the image
    text1 = "Radius of Curvature = "+str(int(mean_curvR))+"m"
    if offset < 0:
        text2 = "Vehicle is "+str(round(-offset, 2))+"m Left of Center"
    else:
        text2 = "Vehicle is "+str(round(offset, 2))+"m Right of Center"

    pos_1 = (100,50)
    pos_2 = (100,120)
    font = cv2.FONT_HERSHEY_SIMPLEX
    #Print curvature value
    cv2.putText(final_img, text1, pos_1,font,2,(255,255,255),2,cv2.LINE_AA)
    #print offset value
    cv2.putText(final_img, text2, pos_2,font,2,(255,255,255),2,cv2.LINE_AA)

    ##Visualize and save image if needed
    # plt.figure(6)
    # plt.imshow(final_img)
    # plt.title('Lane Detection')
    #
    # #Save Image
    # plt.imsave('output_images/final_res.jpg', final_img)
    # plt.show()

    return final_img



#################
### Load Data ###
#################

#Load the camera calibration variables
dist_pickle = pickle.load( open( "camera_cal/calib_mtx.p", "rb" ) )

cmt = dist_pickle["cmt"]
dist = dist_pickle["dist"]

#Load test image
testimg = im.imread('test_images/straight_lines2.jpg')

# #Perform lane detection
# final_img = Lane_detection(testimg,cmt,dist)

##################
### Load Video ###
##################

in_clip = VideoFileClip('test_vids/project_video.mp4')

output_path = 'output_vids/project_video.mp4'

#The output video
out_clip = in_clip.fl_image(Lane_detection)

#Save Processed video
out_clip.write_videofile(output_path, audio=False)
