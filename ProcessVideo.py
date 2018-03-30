
#------------------------------------------------------------------------------
import os
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip
#from IPython.display import HTML

#------------------------------------------------------------------------------
import MachineVision as Mv
from MachineVision import ImageGeometric as ImgGeo
from MachineVision import LaneDetector

#==============================================================================
# Calibrate Camera
ObjPoints = [] # 3D points in real world space
ImgPoints = [] # 2D points in image plane
CornerImages = []

files = os.listdir("camera_cal/")
for file in files:
    if file.startswith("calibration") and file.endswith(".jpg"):
        image = mpimg.imread("camera_cal/" + file) #image in RGB mode
        ImgGeo.draw_chessboard_corners(image, ObjPoints, ImgPoints, CornerImages, 9, 6 )

#==============================================================================
LaneDtct = LaneDetector()

def process_video_image(image):
    
    undistorted_image = ImgGeo.calibrate_and_undistort(image, ObjPoints, ImgPoints)

    mask_gradient = Mv.get_mask_gradient_thresholds(undistorted_image)
    mask_color = Mv.get_mask_color_threshold(undistorted_image)
    mask_gradcolor = np.zeros_like(mask_gradient)
    mask_gradcolor[(mask_gradient==1) | (mask_color==1)] = 1
    binary_image = mask_gradcolor
    
    binary_warped_image, Matrix, Matrix_inverse, src, dst = ImgGeo.warp_image_to_top_down_view(binary_image)
    
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

    return lane_final_image
    
#==============================================================================
if not os.path.exists("test_videos_output"):
    os.makedirs("test_videos_output")
    
white_output = "./project_video_lane_marked.mp4"

clip1 = VideoFileClip("./project_video.mp4")
white_clip = clip1.fl_image(process_video_image)
white_clip.write_videofile(white_output, audio=False)
