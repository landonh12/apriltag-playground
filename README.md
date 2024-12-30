# AprilTag Playground

## File Structure

src/
    PoseEstimator.py contains the Pose Estimator class
    TagPoseKalmanFilter.py contains a Kalman Filter class

resources/
    field drawing PNGs, tools, etc

calibration/
    cap_calib_image.py provides a way to capture calibration images and place them in the calib_images folder.
    calibrate_camera.py provides a way to generate intrinsic camera parameters for use with the AprilTag3 library.

scratch/
    Scratch / test code.

test_class.py implements localization using the PoseEstimator and TagPoseKalmanFilter classes.

## PoseEstimator class

Instructions for using the PoseEstimator class:

Preparing the PoseEstimator class:
1. Calibrate your camera by using the files in the calibration/ folder to generate intrinsic camera parameters.
2. Change camera_params to use your generated parameters.
3. Find the resolution of your camera an dset res_width and res_height.
4. Fill out the tag_coordinates array with field coordinates of tags in inches.
5. Set up cap.appends for the amount of cameras you are using.

Using the PoseEstimator class:
1. Instantiate the class
2. Run get_camera_pose(2).

get_camera_pose returns an array of x,y,z,roll,pitch,yaw field coordinates.

## Current issues

1. The Kalman filter needs some work. More work needs to be understood how to harness it fully.
2. get_camera_pose only returns pose for one camera. In multiple camera setups, we would like to return a fused pose from the class itself rather than relying on the user to do that outside of the class.