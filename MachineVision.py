
#==============================================================================
# Import of Libraries
#==============================================================================
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


#class MaskThreshold:
#==============================================================================
# Functions to Apply Mask of Color Thresholds
#==============================================================================
RED_THRESHOLD = (200,255)
GREEN_THRESHOLD = (200,255)
HUE_THRESHOLD = (10,90)
SATURATION_THRESHOLD = (100,255)
#--------------------------------------------------------------------------
def get_thresh_mask(image, thresh=(0,255)):
    #Create a copy of the image and apply the threshold 
    #ones where threshold is met, zeros otherwise
    mask = np.zeros_like(image)
    mask[(image > thresh[0]) & (image <= thresh[1])] = 1
    return mask
    
#--------------------------------------------------------------------------
def get_mask_color_threshold(image):
    R_channel = image[:,:,0]
    G_channel = image[:,:,1]
    B_channel = image[:,:,2]
    R_mask = get_thresh_mask(R_channel, thresh=RED_THRESHOLD)
    G_mask = get_thresh_mask(G_channel, thresh=GREEN_THRESHOLD)
    
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    H_channel = hls[:,:,0]
    L_channel = hls[:,:,1]
    S_channel = hls[:,:,2]
    assert(type(H_channel[0][0]) == np.uint8)
    H_mask = get_thresh_mask(H_channel, thresh=HUE_THRESHOLD)
    S_mask = get_thresh_mask(S_channel, thresh=SATURATION_THRESHOLD)
    
    combined_thresholds = np.zeros_like(S_mask)
    combined_thresholds[( ((R_mask==1)&(G_mask==1)) | ((H_mask==1)&(S_mask==1)) )] = 1
    return combined_thresholds
    
#==============================================================================
# Functions to Apply Mask of Gradient Thresholds
#==============================================================================
SOBEL_KERNAL_X = 7
SOBEL_KERNAL_Y = 7
SOBEL_KERNAL_MAG = 7
SOBEL_KERNAL_DIR = 7
X_THRESHOLD = (40, 210)
Y_THRESHOLD = (255, 255) #Y_THRESHOLD unused 
MAG_THRESHOLD = (40, 210)
DIR_THRESHOLD = (1.0, 1.3)

#--------------------------------------------------------------------------
def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
    # Note 1: calling your function with orient='x', thresh_min=5, thresh_max=100
    # Note 2: Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your grad_binary image
    """
    """Calculate directional gradient"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a binary image of ones where threshold is met, zeros otherwise
    grad_binary = get_thresh_mask(scaled_sobel, thresh)
    return grad_binary
    
#--------------------------------------------------------------------------
def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):
    """
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your mag_binary image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = get_thresh_mask(gradmag, thresh)
    return mag_binary    
    
#--------------------------------------------------------------------------
def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your dir_binary image
    """
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create a binary image of ones where threshold is met, zeros otherwise
    dir_binary = get_thresh_mask(absgraddir, thresh)
    return dir_binary
    
#--------------------------------------------------------------------------
def get_mask_gradient_thresholds(image):
    # Choose a Sobel kernel size
    #ksize = 3 # Choose a larger odd number to smooth gradient measurements
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=SOBEL_KERNAL_X, thresh= X_THRESHOLD)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=SOBEL_KERNAL_Y, thresh= Y_THRESHOLD)
    mag_binary = mag_thresh(image,             sobel_kernel=SOBEL_KERNAL_MAG, thresh= MAG_THRESHOLD)
    dir_binary = dir_thresh(image,           sobel_kernel=SOBEL_KERNAL_DIR, thresh= DIR_THRESHOLD)
    combined_thresholds = np.zeros_like(dir_binary)
    #combined_thresholds[((gradx==1)|(mag_binary == 1)) & ((grady == 1)|(mag_binary == 1))] = 1
    #combined_thresholds[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined_thresholds[ ((gradx == 1)) | ((mag_binary == 1)&(dir_binary == 1)) ] = 1
    return combined_thresholds

#-------------------------------------------------------------------------------
    '''
    imshape = warped_img.shape
    vertices = np.array([[(200, imshape[0]), (200, 0), (imshape[1] - 200, 0), 
                      (imshape[1]-200, imshape[0])]], dtype=np.int32)
    '''
def crop_image_in_region(img, vertices):
    mask = np.zeros_like(img)
    """
    vertices = np.array([[(580,450),
                          (700,450), 
                          (145,670), 
                          (1135,670)]], 
                          dtype=np.int32)
    """
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    cropped_image = cv2.bitwise_and(img, mask)
    
    return cropped_image


class ImageGeometric:
    #==============================================================================
    # Functions for Camera Calibration, Image Undistortion & Perspective Transformation
    #==============================================================================
    #ObjPoints = [] # 3D points in real world space
    #ImgPoints = [] # 2D points in image plane
    #CornerImages = []

    #-------------------------------------------------------------------------------
    def draw_chessboard_corners(image, object_points, image_points, cornerImages, nx=9, ny=6):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((ny*nx,3),np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) # x, y coordinate
        
        # Arrays to store object points and image points from all the images.
        #object_points = [] # 3d points in real world space
        #image_points = [] # 2d points in image plane.
        #cornerImages = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret == True:
            object_points.append(objp)
            image_points.append(corners)
            cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            cornerImages.append(image)
        else:
            cornerImages.append(image) #append(None)

    #-------------------------------------------------------------------------------
    def calibrate_and_undistort(image, objpoints, imgpoints):
        """
        Function that takes an image, object points, and image points
        performs the camera calibration, image distortion correction and 
        returns the undistorted image
        """
        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        undist = cv2.undistort(image, mtx, dist, None, mtx)
        return undist
        
        
    #-------------------------------------------------------------------------------
    def warp_image_to_top_down_view(image):

        x_offset = 230
        y_offset = 20
        
        height,width = image.shape[:2]

        #Coordinate Format: [top left, top right, bottom left, bottom right]
        src = np.float32([(580,450),
                          (700,450), 
                          (145,670), 
                          (1135,670)])
                          
        dst = np.float32([(x_offset,        y_offset),
                          (width-x_offset,  y_offset),
                          (x_offset,        height-y_offset),
                          (width-x_offset,  height-y_offset)])
                          
        # Given src and dst points, calculate the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src, dst)
        matrix_inverse = cv2.getPerspectiveTransform(dst, src)
        # Warp an image using the perspective transform matrix to make it a top-down view
        warped_image = cv2.warpPerspective(image, matrix, (width,height), flags=cv2.INTER_LINEAR)
        
        #vertices = np.array([[(70, 0),
        #                      (width - 70, 0), 
        #                      (70, height), 
        #                      (width - 70, height)]], 
        #                      dtype=np.int32)
        #warped_image = crop_image_in_region(warped_image, vertices)
        
        return warped_image, matrix, matrix_inverse, src, dst

#==============================================================================
# Class for Lane Line 
#==============================================================================
class Line:

    MAX_COUNT_RECORD = 8
    
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        
        # x values of the last n fits of the line
        ###self.recent_xfitted = [] #collections.deque(10*[0.0, 0.0, 0.0], 10)
        
        #average x values of the fitted line over the last n iterations
        ###self.bestx = None
        
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        
        #radius of curvature of the line in some units
        #self.radius_of_curvature = None 
        
        #distance in meters of vehicle center from the line
        #self.line_base_pos = None 
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        
        #x values for detected line pixels
        ###self.allx = None  
        #y values for detected line pixels
        ###self.ally = None
    
    def update_status(self, fit, inds):
    
        if fit is not None:
            
            if self.best_fit is not None:
                self.diffs = abs(fit-self.best_fit)
            else:
                self.diffs = np.array([0,0,0], dtype='float')
                
            # x = 0.001y^2 + 0.1y + 100
            if ((self.diffs[0] < 0.001) or (self.diffs[1] < 1.0) or (self.diffs[2] < 100.0)):
                self.detected = True
                self.current_fit.append(fit)
                if len(self.current_fit) > self.MAX_COUNT_RECORD:
                    self.current_fit = self.current_fit[(len(self.current_fit)-self.MAX_COUNT_RECORD):self.MAX_COUNT_RECORD]
            else:
                self.detected = False
            
            self.best_fit = np.average(self.current_fit, axis=0)
            
        else:
            self.detected = False
            
#==============================================================================
# Class for Lane Detection 
#==============================================================================
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Set number of sliding windows
NUM_SLIDING_WINDOWS = 9

class LaneDetector:

    def __init__(self):
        self.LeftLine = Line();
        self.RightLine = Line();

    #-------------------------------------------------------------------------------
    def find_lane_base(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        return leftx_base, rightx_base, histogram

    #-------------------------------------------------------------------------------
    def scan_entire_image_for_lane(self, binary_warped):
        
        leftx_base, rightx_base, histogram = self.find_lane_base(binary_warped)

        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/NUM_SLIDING_WINDOWS)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        rectangles = []
        
        # Step through the windows one by one
        for window in range(NUM_SLIDING_WINDOWS):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            # save the windows to be drawn on the visualization image (using cv2.rectangle() )
            rectangles.append((win_xleft_low, win_xright_low, win_y_low, \
                              win_xleft_high, win_xright_high, win_y_high))
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high), (0,255,0), 2)
            #cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high), (0,255,0), 2)
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        left_detected = True
        right_detected = True
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        return  left_fit, right_fit, \
                left_lane_inds, right_lane_inds, \
                rectangles, histogram
        
    #-------------------------------------------------------------------------------
    def scan_partial_image_for_lane(self, binary_warped, prev_left_fit, prev_right_fit):
        
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_lane_inds = (  (nonzerox > (prev_left_fit[0] * (nonzeroy**2) + 
                                         prev_left_fit[1] * nonzeroy + 
                                         prev_left_fit[2] - margin)) 
                                       & 
                            (nonzerox < (prev_left_fit[0] * (nonzeroy**2) + 
                                         prev_left_fit[1] * nonzeroy + 
                                         prev_left_fit[2] + margin))
                         )

        right_lane_inds = ( (nonzerox > (prev_right_fit[0] * (nonzeroy**2) + 
                                         prev_right_fit[1] * nonzeroy + 
                                         prev_right_fit[2] - margin)) 
                                        & 
                            (nonzerox < (prev_right_fit[0] * (nonzeroy**2) + 
                                         prev_right_fit[1] * nonzeroy + 
                                         prev_right_fit[2] + margin))
                          )
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        return left_fit, right_fit, left_lane_inds, right_lane_inds
        
    #-------------------------------------------------------------------------------
    def find_lane_lines(self, binary_warped):
        if 1 == 1:
        #if( (self.LeftLine.detected == False) or (self.RightLine.detected == False) or \
        #    (self.LeftLine.best_fit is None) or (self.RightLine.best_fit is None) ):
            
            left_fit, right_fit, left_lane_inds, right_lane_inds, rectangles, histogram = \
                    self.scan_entire_image_for_lane(binary_warped)
                                        
        else:
            left_fit, right_fit, left_lane_inds, right_lane_inds = \
                    self.scan_partial_image_for_lane(binary_warped, self.LeftLine.best_fit, self.RightLine.best_fit)
        
        self.LeftLine.update_status(left_fit, left_lane_inds)
        self.RightLine.update_status(right_fit, right_lane_inds)
        
        # Sanity Check
        # 1) Valid Lane Width: [500, 800] pixels
        # 2) 
        return left_fit, right_fit, left_lane_inds, right_lane_inds
    #-------------------------------------------------------------------------------
    
#==============================================================================
# Functions to read images and draw images 
#==============================================================================
def show_histogram_for_lane_base(histogram):
    plt.plot(histogram)
    plt.xlim(0, 1280)

#------------------------------------------------------------------------------
def get_ploty_of_detected_lane(binary_warped, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return ploty, left_fitx, right_fitx
    
#------------------------------------------------------------------------------
def visualize_detected_lane(binary_warped, rectangle_group, left_fit, right_fit, left_lane_inds, right_lane_inds, ImageIndex, boIsWindowSliceView):

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
        
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate x and y values for plotting
    ploty, left_fitx, right_fitx = get_ploty_of_detected_lane(binary_warped, left_fit, right_fit)
    
    if boIsWindowSliceView == 1: 
    
        for index in range(NUM_SLIDING_WINDOWS):
            rectangle = rectangle_group[index]
            # Draw rectangle windows on the visualization image
            cv2.rectangle(out_img,(rectangle[0], rectangle[2]), (rectangle[3], rectangle[5]), (0,255,0), 2)
            cv2.rectangle(out_img,(rectangle[1], rectangle[2]), (rectangle[4], rectangle[5]), (0,255,0), 2)
            out_img = out_img.astype(np.uint8)
            assert(out_img.shape == (720, 1280, 3))
            assert(out_img[0, 0, :].dtype == np.uint8)
    
    else:
    
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                  ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                  ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        window_img = np.zeros_like(out_img)
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        out_img = out_img.astype(np.uint8)
        assert(out_img.shape == (720, 1280, 3))
        assert(out_img[0, 0, :].dtype == np.uint8)
            
    lane_mark_img = plt.figure(figsize=(12,9))
    subplot = lane_mark_img.add_subplot(1,1,1)
    subplot.imshow(out_img)
    subplot.plot(left_fitx, ploty, color='yellow')
    subplot.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    #plt.show()

    if not os.path.exists("output_images/lane_marked_images/"):
        os.makedirs("output_images/lane_marked_images/")
    
    if(boIsWindowSliceView == 1):
        lane_mark_img.savefig("output_images/lane_marked_images/" + str(ImageIndex+1).zfill(2) + "_03LaneMarkWindow" + ".jpg")
        lane_mark_img.savefig("output_images/all/" + str(ImageIndex+1).zfill(2) + "_03LaneMarkWindow" + ".jpg")
    else:
        lane_mark_img.savefig("output_images/lane_marked_images/" + str(ImageIndex+1).zfill(2) + "_04LaneMarkPoly" + ".jpg")
        lane_mark_img.savefig("output_images/all/" + str(ImageIndex+1).zfill(2) + "_04LaneMarkPoly" + ".jpg")
    
    plt.close() 
    
    return lane_mark_img;
    
#------------------------------------------------------------------------------
def draw_lane_on_org_image(undist, warped, ploty, left_fitx, right_fitx, left_lane_inds, right_lane_inds, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    #####print(color_warp.shape)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
        
    color_warp[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    color_warp[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    result = result.astype(np.uint8)
    
    assert(result.shape == (720, 1280, 3))
    assert(result[0, 0, :].dtype == np.uint8)
    
    return result;

#------------------------------------------------------------------------------
def show_data_on_org_image(org_img, left_curverad, right_curverad, center_dist, boShowMeanCurveRadius):
    processed_img = np.copy(org_img)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    
    if boShowMeanCurveRadius == 0:
        text = 'Left Curve Radius: ' + '{:04.2f}'.format(left_curverad/1000.0) + 'km'
        cv2.putText(processed_img, text, (30,70), font, fontScale=1.2, color=(190,255,190), thickness=2, lineType=cv2.LINE_AA)
        
        text = 'Right Curve Radius: ' + '{:04.2f}'.format(right_curverad/1000.0) + 'km'
        cv2.putText(processed_img, text, (30,110), font, fontScale=1.2, color=(190,255,190), thickness=2, lineType=cv2.LINE_AA)
        
    else: #elif boShowMeanCurveRadius == 1:
        curvrad = np.mean([left_curverad, right_curverad])
        text = 'Curve Radius: ' + '{:04.2f}'.format(curvrad/1000.0) + 'km'
        cv2.putText(processed_img, text, (30,110), font, fontScale=1.2, color=(190,255,190), thickness=2, lineType=cv2.LINE_AA)
        
    direction = ""
    if center_dist < 0.0:
        direction = "right"
    elif center_dist > 0.0:
        direction = "left"
    
    text = 'Distance from Lane Center: ' + '{:04.3f}'.format(abs(center_dist)) + 'm ' + direction
    cv2.putText(processed_img, text, (30,150), font, fontScale=1.2, color=(190,255,190), thickness=2, lineType=cv2.LINE_AA)
    
    return processed_img
    
#==============================================================================
# Functions to measure curvature and center offset 
#==============================================================================
def check_lane_sanity(binary_warped, left_fit, right_fit):
    
    lane_valid = True
    height = binary_warped.shape[0]
    left_intercept_bottom = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
    right_intercept_bottom = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
    left_intercept_top = left_fit[0]*0**2 + left_fit[1]*0 + left_fit[2]
    right_intercept_top = right_fit[0]*0**2 + right_fit[1]*0 + right_fit[2]
    
    width_bottom = right_intercept_bottom - left_intercept_bottom
    width_top = right_intercept_top - left_intercept_top
    
    if ( (width_bottom <= 500.0)   
         or (width_bottom >= 800.0)
         or (width_top <= 500.0) 
         or (width_top >= 800.0) ):
        lane_valid = False
    
    #####print(lane_valid)
    #####print(width_bottom)
    #####print(width_top)
    
    return lane_valid, width_bottom, width_top, left_intercept_bottom, right_intercept_bottom
    
#------------------------------------------------------------------------------
def measure_curvature_n_center_offset(binary_warped, left_fit, right_fit, left_lane_inds, right_lane_inds):

    left_curverad, right_curverad, center_dist = (0, 0, 0)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Generate some fake data to represent lane-line pixels
    height = binary_warped.shape[0]
    ploty = np.linspace(0, height-1, num=height)# to cover same y-range as image

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Define conversions in x and y from pixels space to meters
    """
    Convert our x and y values to real world space
    Assume: the lane is about 30 meters long and 3.7 meters wide.
             each section of dashed lane line is 3 meters
    """
    ym_per_pix = 30/700 # meters per pixel in y dimension
    xm_per_pix = 3.7/600 # meters per pixel in x dimension    
    
    if len(leftx) != 0 and len(rightx) != 0:
    
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
    
    # offset of the lane center from the center of the image is distance from the center of the lane 
    if (left_fit is not None) and (right_fit is not None):
        car_center = binary_warped.shape[1]/2
        left_fitx_offset = left_fit[0]*height**2 + left_fit[1]*height + left_fit[2]
        right_fitx_offset = right_fit[0]*height**2 + right_fit[1]*height + right_fit[2]
        lane_center = (left_fitx_offset + right_fitx_offset) /2
        center_dist = (lane_center - car_center) * xm_per_pix
    
    return left_curverad, right_curverad, center_dist
    
#------------------------------------------------------------------------------
    
    