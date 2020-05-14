import matplotlib.image as im
import numpy as np
import cv2
import glob
import pickle


def getCalibrationMatrix (imgs_path,nx,ny):

    #Create listst to hold the corner points for images and real coordinates

    objpoints = []
    imgpoints = []

    #Construct the real life 3D points meshgrid for x,y and z = 0 for all points (assuming flat surface)

    objp = np.zeros(((nx*ny),3),np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)


    #Obtain points for each image
    for path in imgs_path:

        img = im.imread(path)

        #Detect corners
        res,corners = cv2.findChessboardCorners(img,(nx,ny))

        if res : #Corners found
            objpoints.append(objp)
            imgpoints.append(corners)

    #Extract the size of the last image
    size = (img.shape[1],img.shape[0])

    #Compute calibration Matrix
    _,cmt,dist,_,_ = cv2.calibrateCamera(objpoints,imgpoints,size,None,None)

    return (cmt,dist)



####################################################
### Get Camera Calibration Matrix (ONLY ONCE)#######
####################################################
#grid size
nx = 9
ny = 6


path = glob.glob('camera_cal/calibration*.JPG')
cmt,dist = getCalibrationMatrix(path,nx,ny)

# Save the camera calibration result for later use
dist_pickle = {}
dist_pickle["cmt"] = cmt
dist_pickle["dist"] = dist
dist_pickle["dx"] = nx
dist_pickle["dy"] = ny
pickle.dump( dist_pickle, open( "camera_cal/calib_mtx.p", "wb" ) )


#Print that the program is done
print("Camera Matrix has been saved successfully !")
