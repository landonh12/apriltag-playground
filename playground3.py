# TODO: This is so messy, need to refactor..

import cv2
import time
from pupil_apriltags import Detector
import numpy as np
import matplotlib.pyplot as plt

# These are the poses of the tags on my wall at the moment
# 1.25 meters from the floor, 0.142785 meters to the left/right of the center point between each tag, 0 z (depth)
id_2_pose = np.array([-0.1397, -1.25095, 0], dtype=np.float32)
id_3_pose = np.array([0.1397, -1.25095, 0], dtype=np.float32)

# Camera intrinsics
# Generated from calibrate_camera.py - Use cap_calib_image.py to capture images
# TODO: redo this, idk if this is entirely accurate. camera center is a bit off. 
# Ideally camera center cx,cy is 640, 400 (half of 1280x800)
# TODO: maybe read from a yaml file or something for this for easy editing
fx, fy, cx, cy = (941.287545323049, 936.9720277635304, 660.6219235101386, 413.5921687946573)
camera_matrix = np.array([[fx, 0., cx],[0., fy, cy],[0., 0., 1.]], dtype=np.float32)

# Constants
TAG_SIZE = 0.1651 # Width in meters (of the tag)

### Function definitions ###

# Draw lines around tag to show it's detected (for debugging)
def draw_lines(frame, results):
    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(frame, tagFamily, (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Rotation matrix is not in the format I like
# I like roll pitch yaw in degrees
def reshape_rot(rot):
    # TODO: Need to test the below after the whole tag->camera pose code addition...
    #in - pitch yaw roll, radians
    #out - roll pitch yaw, degrees
    return np.array([np.rad2deg(rot[2][0]), np.rad2deg(rot[1][0]), np.rad2deg(rot[0][0])])


# Just a function to print out results so I can easily view data
# TODO: break this up into a "get camera pose" function and a "print results" function.
# The print results function is doing all the work right now lol
def print_results(results):
    rot = [np.zeros(3), np.zeros(3)]
    trans = [np.zeros(3), np.zeros(3)]
    id = -1
    if(len(results) > 0):
        for i in range(len(results)):
            id = results[i].tag_id

            # Good lord, I have no idea what I'm doing here.
            # So, the "pose" that we get from the AprilTag library is the "tag pose" in camera coordinates?
            # We want to get camera pose in "object coordinates" or "world coordinates"
            # So to do that, according to https://stackoverflow.com/questions/18637494/camera-position-in-world-coordinate-from-cvsolvepnp?rq=2
            # We need to get the transpose of the 3x3 rotation matrix, R'
            # Then we need to multiply -R' by the translation vector
            # That will give us the camera pose?
            # I guess we just take the transpose of R and plug that into cv2.Rodrigues?
            # We will test when we get home...
            rot_vec = results[i].pose_R
            trans_vec = results[i].pose_t

            rot_vec = rot_vec.transpose()
            trans_vec = -rot_vec @ trans_vec

            rodrigues = np.array(rot_vec)
            #print(rodrigues)
            if(id == 2):
                rot[0], _ = cv2.Rodrigues(rodrigues)
                rot[0] = reshape_rot(rot[0])
            if(id == 3):
                rot[1], _ = cv2.Rodrigues(rodrigues)
                rot[1] = reshape_rot(rot[1])

            if(id == 2):
                trans[0] = np.array([trans_vec[0][0]+id_2_pose[0], trans_vec[1][0]+id_2_pose[1], trans_vec[2][0]+id_2_pose[2]])
            if(id == 3):
                trans[1] = np.array([trans_vec[0][0]+id_3_pose[0], trans_vec[1][0]+id_3_pose[1], trans_vec[2][0]+id_3_pose[2]])
            print("Tag ID: " + str(id))
            print("Pose Error: " + str(results[i].pose_err))
        trans_avg = (trans[0] + trans[1]) / 2.0
        rot_avg = (rot[0] + rot[1]) / 2.0
        print("Rotation vector: " + str(rot_avg))
        print("Translation vector: " + str(trans_avg) + "\n")

    # Return data for CSV writing        
    return rot, trans, id


# TODO: BARF
def save_to_csv(poses):
    with open("data.csv", "w+") as f:
        f.write("timestamp,id,x,y,z,roll,pitch,yaw\n")
        for i in range(len(poses)):
            t = str(poses[i][0])
            f_id = str(poses[i][1])
            x = str(poses[i][2])
            y = str(poses[i][3])
            z = str(poses[i][4])
            roll = str(poses[i][5])
            pitch = str(poses[i][6])
            yaw = str(poses[i][7])
            c = ","
            # This line makes me want to.. $*&/@
            f.write(t + c + f_id + c + x + c + y + c + z + c + roll + c + pitch + c + yaw + "\n")
    f.close()

def set_camera_params(cap):
    # Set exposure. 1 is manual, 3 is auto
    print("Setting exposure...")
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    # 50 is a good number for decent lighting.
    # Need to go lower if you want less blur.
    # 10 or below - might need lighting
    cap.set(cv2.CAP_PROP_EXPOSURE, 10)

    # Print exposure after setting
    print("Auto Exposure: " + str(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)))
    print("Absolute Exposure: " + str(cap.get(cv2.CAP_PROP_EXPOSURE)))

    # Set resolution to 1280x800 - max camera res, we need all the res we can get
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    # Set to MJPG encoding - YUYV does not support 100fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# TODO: God please make a main function
# Prayers answered
def main():
    # open video0
    cap = cv2.VideoCapture(0)
    time.sleep(1)
    if(not cap.isOpened()):
        exit

    set_camera_params(cap)

    # This is old, from apriltag2 library. Options now go in Detector class constructor
    #options = apriltag.DetectorOptions(families="tag36h11")
    detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1)

    pose = []

    while(True):
        if __debug__:
            t1 = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Convert to GRAY - can we just capture as GRAY?
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # results is an array with the following shape:
        results = detector.detect(gray, estimate_tag_pose=True, camera_params=[fx,fy,cx,cy], tag_size=TAG_SIZE)
        frame = draw_lines(frame, results)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if __debug__:
            t2 = time.time()
        #print("Runtime: " + str(t2-t1))
        rot, trans, r_id = print_results(results)
        #pose.append([t2, r_id, trans[0], trans[1], trans[2], rot[0], rot[1], rot[2]])

    #save_to_csv(pose)

    # When everything done, release the capture
    cap.release() 
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()