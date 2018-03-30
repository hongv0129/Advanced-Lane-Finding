
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import MachineVision as Mv
from MachineVision import ImageGeometric  as ImgGeo
from MachineVision import LaneDetector
#from MachineVision import MaskThreshold  as MskThresh

#==============================================================================
# [1] Camera Calibration 
#==============================================================================
ObjPoints = [] # 3D points in real world space
ImgPoints = [] # 2D points in image plane
CornerImages = []
UndistortedChessboards = []

OriginImages = []
UndistortedImages = []
WarpedImages = []
BinaryImages = []
WarpedBinaryImages = []
LaneImages = []

LaneMarkWindows = []
LaneMarkPolys = []

#-------------------------------------------------------------------------
# Calibrate Camera with chessboard pattern
files = os.listdir("camera_cal/")
for file in files:
    if file.startswith("calibration") and file.endswith(".jpg"):
        image = mpimg.imread("camera_cal/" + file) #image in RGB mode
        ImgGeo.draw_chessboard_corners(image, ObjPoints, ImgPoints, CornerImages, 9, 6 )
print("Total number of chessboard images : ", len(CornerImages))
print("Shape of corner images : ", CornerImages[0].shape)

if not os.path.exists("output_images/chessboard_corner/"):
    os.makedirs("output_images/chessboard_corner/")

for index in range(0, len(CornerImages), 1):
    mpimg.imsave("output_images/chessboard_corner/ChessboardCorner" + str(index+1) + ".jpg", CornerImages[index])

#-------------------------------------------------------------------------
# Undistort a chessboard image (example)
    image = mpimg.imread("camera_cal/calibration1.jpg") #image in RGB mode
    undistorted_chessboard = ImgGeo.calibrate_and_undistort(image, ObjPoints, ImgPoints)

f, ax = plt.subplots(1, 2, figsize=(10.8, 3))
f.tight_layout()
ax[0].imshow(image)
ax[0].set_title('Original Image', fontsize=16)
ax[1].imshow(undistorted_chessboard)
ax[1].set_title('Undistorted Image', fontsize=16)
plt.subplots_adjust(left=0.06, right=0.97, top=0.98, bottom=0.02)
#plt.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.1)

#==============================================================================
# [2] Process Images in a Pipeline (Step by Step)
#==============================================================================
# [2.1] Undistort road images based on the calibrated Camera
files = os.listdir("test_images/")
for file in files:
    image = mpimg.imread("test_images/" + file) #image in RGB mode
    OriginImages.append(image)
    undistorted_image = ImgGeo.calibrate_and_undistort(image, ObjPoints, ImgPoints)
    UndistortedImages.append(undistorted_image)
assert(len(OriginImages) == len(UndistortedImages))
print("Total number of undistorted images : ", len(UndistortedImages))
    
if not os.path.exists("output_images/undistorted_images/"):
    os.makedirs("output_images/undistorted_images/")

for index in range(0, len(UndistortedImages), 1):
    mpimg.imsave("output_images/undistorted_images/" + str(index+1).zfill(2) + "_00Undistorted" + ".jpg", UndistortedImages[index])
    mpimg.imsave("output_images/all/" + str(index+1).zfill(2) + "_00Undistorted" + ".jpg", UndistortedImages[index])

#-------------------------------------------------------------------------
f, ax = plt.subplots(1, 2, figsize=(10.8, 3))
f.tight_layout()
ax[0].imshow(OriginImages[6])
ax[0].set_title('Original Image', fontsize=16)
ax[1].imshow(UndistortedImages[6])
ax[1].set_title('Undistorted Image', fontsize=16)
plt.subplots_adjust(left=0.06, right=0.97, top=0.98, bottom=0.02)
#plt.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.1)

#f, ax = plt.subplots(2, 2, figsize=(24, 9))
#f.tight_layout()
#ax[0,0].imshow(OriginImages[0])
#ax[0,0].set_title('Original Image', fontsize=16)
#ax[0,1].imshow(UndistortedImages[0])
#ax[0,1].set_title('Undistorted Image', fontsize=16)
#ax[1,0].imshow(OriginImages[1])
#ax[1,0].set_title('Original Image', fontsize=16)
#ax[1,1].imshow(UndistortedImages[1])
#ax[1,1].set_title('Undistorted Image', fontsize=16)
#plt.subplots_adjust(left=0.0, right=1.0, top=0.7, bottom=0.0)
#plt.show()
#==============================================================================
# [2.2] Create thresholded binary image from the undistort road images
for index in range(0, len(UndistortedImages), 1):
    mask_gradient = Mv.get_mask_gradient_thresholds(UndistortedImages[index])
    mask_color = Mv.get_mask_color_threshold(UndistortedImages[index])
    mask_gradcolor = np.zeros_like(mask_gradient)
    mask_gradcolor[(mask_gradient==1) | (mask_color==1)] = 1
    BinaryImages.append(mask_gradcolor)
    
if not os.path.exists("output_images/binary_images/"):
    os.makedirs("output_images/binary_images/")

for index in range(0, len(BinaryImages), 1):
    mpimg.imsave("output_images/binary_images/" + str(index+1).zfill(2) + "_01Binary" + ".jpg", BinaryImages[index])
    mpimg.imsave("output_images/all/" + str(index+1).zfill(2) + "_01Binary" + ".jpg", BinaryImages[index])
    
#-------------------------------------------------------------------------
f, ax = plt.subplots(1, 2, figsize=(10.8, 3))
f.tight_layout()
ax[0].imshow(UndistortedImages[6])
ax[0].set_title('Undistorted Image', fontsize=16)
ax[1].imshow(BinaryImages[6])
ax[1].set_title('Thresholded Binary Image', fontsize=16)
plt.subplots_adjust(left=0.06, right=0.97, top=0.98, bottom=0.02)
#plt.show()

#==============================================================================
# [2.3] Warp the undistort road images & thresholded binary images
for index in range(0, len(UndistortedImages), 1):
    warped_image, Matrix, Matrix_inverse, src, dst = \
                        ImgGeo.warp_image_to_top_down_view(UndistortedImages[index])
    WarpedImages.append(warped_image)
    
if not os.path.exists("output_images/warped_images/"):
    os.makedirs("output_images/warped_images/")

for index in range(0, len(WarpedImages), 1):
    mpimg.imsave("output_images/warped_images/" + str(index+1).zfill(2) + "_01Warped" + ".jpg", WarpedImages[index])
    mpimg.imsave("output_images/all/" + str(index+1).zfill(2) + "_01Warped" + ".jpg", WarpedImages[index])
    
#-------------------------------------------------------------------------
f, ax = plt.subplots(1, 2, figsize=(10.8, 3))
f.tight_layout()

ax[0].imshow(UndistortedImages[6])
ax[0].set_title('Undistorted Image', fontsize=16)
x = [src[0][0],src[2][0],src[3][0],src[1][0],src[0][0]]
y = [src[0][1],src[2][1],src[3][1],src[1][1],src[0][1]]
ax[0].plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
#ax[0].set_ylim([h,0])
#ax[0].set_xlim([0,w])

ax[1].imshow(WarpedImages[6])
ax[1].set_title('Warped Image', fontsize=16)
x = [dst[0][0],dst[2][0],dst[3][0],dst[1][0],dst[0][0]]
y = [dst[0][1],dst[2][1],dst[3][1],dst[1][1],dst[0][1]]
ax[1].plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
plt.subplots_adjust(left=0.06, right=0.97, top=0.98, bottom=0.02)
#plt.show()

#==============================================================================
for index in range(0, len(BinaryImages), 1):
    warped_binary_image, Matrix, Matrix_inverse, src, dst= ImgGeo.warp_image_to_top_down_view(BinaryImages[index])
    WarpedBinaryImages.append(warped_binary_image)
    
if not os.path.exists("output_images/warped_binary_images/"):
    os.makedirs("output_images/warped_binary_images/")

for index in range(0, len(WarpedBinaryImages), 1):
    mpimg.imsave(fname="output_images/warped_binary_images/" + str(index+1).zfill(2) + "_02WarpedBinary" + ".jpg", arr=WarpedBinaryImages[index], cmap='gray')
    mpimg.imsave(fname="output_images/all/" + str(index+1).zfill(2) + "_02WarpedBinary" + ".jpg", arr=WarpedBinaryImages[index], cmap='gray')
    
#-------------------------------------------------------------------------
f, ax = plt.subplots(1, 2, figsize=(10.8, 3))
f.tight_layout()

ax[0].imshow(BinaryImages[6])
ax[0].set_title('Undistorted Image', fontsize=16)
x = [src[0][0],src[2][0],src[3][0],src[1][0],src[0][0]]
y = [src[0][1],src[2][1],src[3][1],src[1][1],src[0][1]]
ax[0].plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
#ax[0].set_ylim([h,0])
#ax[0].set_xlim([0,w])

ax[1].imshow(WarpedBinaryImages[6])
ax[1].set_title('Warped Image', fontsize=16)
x = [dst[0][0],dst[2][0],dst[3][0],dst[1][0],dst[0][0]]
y = [dst[0][1],dst[2][1],dst[3][1],dst[1][1],dst[0][1]]
ax[1].plot(x, y, color='red', alpha=0.4, linewidth=3, solid_capstyle='round', zorder=2)
plt.subplots_adjust(left=0.06, right=0.97, top=0.98, bottom=0.02)
#plt.show()

#==============================================================================
# [2.4] Detect lane in the warped images, and draw lane onto the original images

LaneDtct00 = LaneDetector()

if not os.path.exists("output_images/final_lane_images/"):
    os.makedirs("output_images/final_lane_images/")
        
for index in range(0, len(WarpedBinaryImages), 1):
    left_fit, right_fit, left_lane_inds, right_lane_inds, rectangles, histogram = \
                                    LaneDtct00.scan_entire_image_for_lane(WarpedBinaryImages[index])
                                    
    LaneMarkWindow = Mv.visualize_detected_lane(WarpedBinaryImages[index], rectangles, \
                                                left_fit, right_fit, \
                                                left_lane_inds, right_lane_inds, index, 1)
    
    LaneMarkPoly = Mv.visualize_detected_lane(WarpedBinaryImages[index], rectangles, \
                                            left_fit, right_fit, \
                                            left_lane_inds, right_lane_inds, index, 0)

    LaneMarkWindows.append(LaneMarkWindow)
    LaneMarkPolys.append(LaneMarkPoly)

    #--------------------------------------------------------------------------
    ploty, left_fitx, right_fitx = Mv.get_ploty_of_detected_lane(WarpedBinaryImages[index], left_fit, right_fit)
    
    lane_valid, width_bottom, width_top, left_intercept_bottom, right_intercept_bottom = \
                                                Mv.check_lane_sanity(WarpedBinaryImages[index], left_fit, right_fit)
    
    lane_image = Mv.draw_lane_on_org_image(UndistortedImages[index], WarpedBinaryImages[index], 
                                            ploty, left_fitx, right_fitx, 
                                            left_lane_inds, right_lane_inds, 
                                            Matrix_inverse)
                                            
    left_curverad, right_curverad, center_dist = Mv.measure_curvature_n_center_offset(WarpedBinaryImages[index], left_fit, 
                                                                                right_fit, left_lane_inds, right_lane_inds)
    
    lane_image = Mv.show_data_on_org_image(lane_image, left_curverad, right_curverad, center_dist, 0)
    
    LaneImages.append(lane_image)
    
    mpimg.imsave("output_images/final_lane_images/" + str(index+1).zfill(2) + "_05LaneFinal" + ".jpg", LaneImages[index])
    mpimg.imsave("output_images/all/" + str(index+1).zfill(2) + "_05LaneFinal" + ".jpg", LaneImages[index])
    
#-------------------------------------------------------------------------
image1 = mpimg.imread("output_images/lane_marked_images/07_03LaneMarkWindow.jpg")
image2 = mpimg.imread("output_images/lane_marked_images/07_04LaneMarkPoly.jpg")
f, ax = plt.subplots(1, 2, figsize=(10.8, 3))
f.tight_layout()
ax[0].imshow(image1)
ax[0].set_title('Lane Detect(sliding Window)', fontsize=16)
ax[1].imshow(image2)
ax[1].set_title('Lane Detect(Polyfit)', fontsize=16)
plt.subplots_adjust(left=0.06, right=0.97, top=0.98, bottom=0.02)
plt.show()

#==============================================================================
# [3] Process Images in a Pipeline
#     (to TEST a combined function for LATER VIDEO Processing )
#==============================================================================
LaneDtct = LaneDetector()

def process_image_pipeline(image):
    
    undistorted_image = ImgGeo.calibrate_and_undistort(image, ObjPoints, ImgPoints)
    mpimg.imsave("output_images/all_pipeline_test/" + str(index+1).zfill(2) + "_00Undistorted" + ".jpg", undistorted_image)

    mask_gradient = Mv.get_mask_gradient_thresholds(undistorted_image)
    mask_color = Mv.get_mask_color_threshold(undistorted_image)
    mask_gradcolor = np.zeros_like(mask_gradient)
    mask_gradcolor[(mask_gradient==1) | (mask_color==1)] = 1
    binary_image = mask_gradcolor
    mpimg.imsave("output_images/all_pipeline_test/" + str(index+1).zfill(2) + "_01Binary" + ".jpg", mask_gradcolor)
    mpimg.imsave("output_images/all_pipeline_test/" + str(index+1).zfill(2) + "_01Color" + ".jpg", mask_color)
    mpimg.imsave("output_images/all_pipeline_test/" + str(index+1).zfill(2) + "_01Gradiant" + ".jpg", mask_gradient)
    
    binary_warped_image, Matrix, Matrix_inverse, src, dst = ImgGeo.warp_image_to_top_down_view(binary_image)
    mpimg.imsave("output_images/all_pipeline_test/" + str(index+1).zfill(2) + "_02WarpedBinary" + ".jpg", binary_warped_image, cmap='gray')
    
    #left_fit, right_fit, left_lane_inds, right_lane_inds = LaneDtct.find_lane_lines(binary_warped_image)
    #ploty, left_fitx, right_fitx = Mv.get_ploty_of_detected_lane(   binary_warped_image, 
    #                                                                LaneDtct.LeftLine.best_fit, 
    #                                                                LaneDtct.RightLine.best_fit)
                                                                    
    left_fit, right_fit, left_lane_inds, right_lane_inds, rectangles, histogram = \
                                                                    LaneDtct.scan_entire_image_for_lane(binary_warped_image)
    
    ploty, left_fitx, right_fitx = Mv.get_ploty_of_detected_lane(   binary_warped_image, 
                                                                    left_fit, 
                                                                    right_fit)
    
    lane_mark_image = Mv.draw_lane_on_org_image(undistorted_image, binary_warped_image, 
                                            ploty, left_fitx, right_fitx, 
                                            left_lane_inds, right_lane_inds, 
                                            Matrix_inverse)
                                            
    left_curverad, right_curverad, center_dist = Mv.measure_curvature_n_center_offset(binary_warped_image, 
                                                                                left_fit, 
                                                                                right_fit, 
                                                                                left_lane_inds, 
                                                                                right_lane_inds)
    
    lane_final_image = Mv.show_data_on_org_image(lane_mark_image, left_curverad, right_curverad, center_dist, 1)
    mpimg.imsave("output_images/all_pipeline_test/" + str(index+1).zfill(2) + "_10PipeLine" + ".jpg", lane_final_image) 

    return lane_final_image
    
#==============================================================================
for index in range(0, len(OriginImages), 1):
    lane_final_image = process_image_pipeline(OriginImages[index])
#    mpimg.imsave("output_images/all_pipeline_test/" + str(index+1).zfill(2) + "_10PipeLine" + ".jpg", lane_final_image) 
