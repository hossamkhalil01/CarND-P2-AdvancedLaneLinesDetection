## Overview

This folder contains the results of each image in [test_images](https://github.com/HossamKhalil-hub01/CarND-P2--AdvancedLaneLinesDetection/tree/master/test_images) folder after being processed by the pipeline.

For each test image you can find a total of five images:


* undistorted: The undistorted image using camera calibration matrix.
* combo_thresh: The combination between gradients and color thresholding.
* BirdEye_view: Perspective Transform for the scene to the bird's eye view.
* Fitted_LaneLines: Shows the lane line fit and the sliding window approach.
* final_res: The result of reflecting the detected lane back to the original view.

**Note:** If you uncommented the visualization section in `Lane_detection()` function found in [Advanced_LaneLines.py](https://github.com/HossamKhalil-hub01/CarND-P2--AdvancedLaneLinesDetection/blob/master/Advanced_LaneLines.py) script, the resulting images will be automatically saved in this directory
