# Advanced Lane Line Detection

## Michael DeFilippo

#### Please see my [project code](https://github.com/mikedef/CarND-Advanced-Lane-Lines/blob/master/advanced-lane-finding-pipeline.ipynb) for any questions regarding implimentation.
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/orig_distorted_img.png "Undistorted"
[image2]: ./output_images/undistort_road_img_sample.png "Road Transformed"
[image3]: ./output_images/sobel_road_img_harder_sample.png "Sobel Example"
[image4]: ./output_images/mag_road_img_sample.png "Mag Example"
[image5]: ./output_images/dir_road_img_sample_kernel3.png "Dir Example"
[image6]: ./output_images/combined_road_img_sample_.png "Combined Example"
[image7]: ./output_images/combined_color_road_img_sample_.png "Combined with color Example"
[image8]: ./output_images/perspective_transform_road_img_sample_.png "perspective transform Example"
[image9]: ./output_images/histogram_road_img_sample_.png "Hist of lane line pixles"
[image10]: ./output_images/slidingWinddow.png "sliding Hist of lane line pixles"
[image11]: ./output_images/FittedLaneLines.png "slidings Hist of lane line pixles"
[image12]: ./output_images/curve_fitting_road_img_sample_.png "curve fitting"
[video14]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first few code cells of the IPython notebook located in "advanced-lane-finding-pipeline.ipynb"  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. I used the functions `cv2.findChessboardCorners()` and `cv2.drawChessboardCorners()`.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I next use the camera calibration matrix to test and undistort on sample road images like the one below: 
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. The code for this step is contained in the Pipeline Helper Functions code cells of the IPython notebook located in "advanced-lane-finding-pipeline.ipynb". Here's an example of my output for this step.

First I explored x and y orientation for Sobel thresholding. See below for an example image:

![alt text][image3]

Next I explored magnitude of the gradient thresholding. See below for an example image:

![alt text][image4]

Next I explored direction of the gradient thresholding. See below for an example image:

![alt text][image5]

Next I combined the above thresholding techniques into a single binary image.

![alt text][image6]

Next I wanted to explore thresholding in the color space. I found that by thresholding in the HLS and the HSV color space I was able to single out the yellow and white lane line pixles. 

![alt text][image7]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Next I transformed the undistorted binary image into a "birds eye view" of the road, such that it only focuses on the lane lines so that the lines apper as you are looking down from above. I used the function to achive such a perspective transform `cv2.getPerspectiveTransform(src,dst)` and the function `cv2.getPerspectiveTransform(dst,src)`
to get the inverse of the transform.

The following source and destination points were used:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 205, 720      | 205, 720        | 
| 1120, 720      | 1120, 720      |
| 745, 480     | 1120, 0      |
| 550, 480      | 205, 0        |

![alt text][image8]

#### 4 and 5. Describe how you identified lane-line pixels and fit their positions with a polynomial:

Next I identified where in the perspective transformed binary image that the lane line pixles were located by taking a histogram of the pixels in the image. 

![alt text][image9]

First I identified where the lane line pixles where by looking at the historgam to show an approximate area to look for lane lines in the image. I was able to identigy all non zero pixels around the histogram peaks using the numpy function  `numpy.nonzero()`. Next I fit a plynomial to each lane using the numpy function `numpy.polyfit()`

Next I performed a sliding window search to find the most likely positions of the 2 lane lines. 

![alt text][image10]

![alt text][image11]

#### 5. Describe how you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Next I was able to calculate the position of the vehicle with respect to center and the radius of curvature for each lane line in meters with the folowing calculations:
    `left_radius = ((1 + (2*left_fit_curve[0]*y_eval*ym_per_pix + left_fit_curve[1])**2)**1.5) / np.absolute(2*left_fit_curve[0])`
    `right_radius = ((1 + (2*right_fit_curve[0]*y_eval*ym_per_pix + right_fit_curve[1])**2)**1.5) / np.absolute(2*right_fit_curve[0])`
    `dist_from_cent = np.abs(center_idx - identified_lanes_center_idx)*xm_per_pix`

![alt text][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/keLANCTYkj4)
![alt text][video14]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
