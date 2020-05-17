## Overview

This folder contains two main components

1. Images to be used for the camera calibration, which is being utilized  by [getCalibrationMatrix.py](https://github.com/HossamKhalil-hub01/CarND-P2--AdvancedLaneLinesDetection/blob/master/getCalibrationMatrix.py) script.

2. [calib_mtx.p](https://github.com/HossamKhalil-hub01/CarND-P2--AdvancedLaneLinesDetection/blob/master/camera_cal/calib_mtx.p) file which contains the calibration matrix and the distortion  coefficients that is the output of **getCalibrationMatrix.py** script.


**Notes:**

- You can change the calibration images with your own camera images just maintain the same images names style (calibrationX.jpg) where X is ordered number from 1 to whatever.

- There is no need to run **getCalibrationMatrix.py** script if you already have the **calib_mtx.p** file
