import cv2
import time
from pupil_apriltags import Detector
import numpy as np
import matplotlib.pyplot as plt

id_2_pose = np.array([1.25095, -0.142785, 0], dtype=np.float32)
id_3_pose = np.array([1.25095, 0.142785, 0], dtype=np.float32)

# Camera intrinsics
fx, fy, cx, cy = (941.287545323049, 936.9720277635304, 660.6219235101386, 413.5921687946573)
camera_matrix = np.array([[fx, 0., cx],[0., fy, cy],[0., 0., 1.]], dtype=np.float32)

# Constants
TAG_SIZE = 0.1651 # Width in meters

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

def reshape_rot(rot):
    #in - pitch yaw roll, radians
    #out - roll pitch yaw, degrees

    return np.array([np.rad2deg(rot[2][0]), np.rad2deg(rot[1][0]), np.rad2deg(rot[0][0])])

def print_results(results):
    rot = np.zeros(3)
    trans = np.zeros(3)
    id = -1
    if(len(results) > 0):
        for i in range(len(results)):
            id = results[i].tag_id
            homography = results[i].homography
            #print(homography)
            #print(type(results[i].corners))
            #corners = results[i].corners
            #corners = corners.astype(np.float32)
            #print(corners)
            #dist_coeffs = np.zeros((5,1))
            #if(id == 2):
            #    retval, rvec, tvec = cv2.solvePnP(id_2_pose, corners, camera_matrix, dist_coeffs)
            #if(id == 3):
            #    retval, rvec, tvec = cv2.solvePnP(id_3_pose, corners, camera_matrix, dist_coeffs)
            #print(rvec)
            #print(tvec)

            #pose = detector.detection_pose(results[i], [fx, fy, cx, cy], tag_size=TAG_SIZE)

            #print(results[i].pose_R)
            #print(results[i].pose_t)

            rvec = results[i].pose_R
            tvec = results[i].pose_t

            rodrigues = np.array(rvec)
            #print(rodrigues)
            rot, _ = cv2.Rodrigues(rodrigues)
            rot = reshape_rot(rot)

            trans = np.array([tvec[0][0], tvec[1][0], tvec[2][0]])
            print("Tag ID: " + str(id))
            #print("Rotation vector shape: " + str(rot.shape))
            print("Pose Error: " + str(results[i].pose_err))
            print("Rotation vector: " + str(rot))
            print("Translation vector: " + str(trans) + "\n")
            #print("Distance (m): " + str(pose[0][2][3]) + "\n")
            #ax.plot(trans[0], trans[1])
            #plt.draw()
            #print("Tvec 1: " + str(pose[0][3]))
    return rot, trans, id

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
            f.write(t + c + f_id + c + x + c + y + c + z + c + roll + c + pitch + c + yaw + "\n")
    f.close()

def set_camera_params(cap):
    # get settings
    #print(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
    #print(cap.get(cv2.CAP_PROP_EXPOSURE))

    # Set exposure
    print("Setting exposure...")
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 50)

    # Print exposure after setting
    print("Auto Exposure: " + str(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)))
    print("Absolute Exposure: " + str(cap.get(cv2.CAP_PROP_EXPOSURE)))

    # Set resolution to 1280x800
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

    # Set to MJPG encoding - YUYV does not support 100fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# open video0
cap = cv2.VideoCapture(0)
time.sleep(1)
if(not cap.isOpened()):
    exit

set_camera_params(cap)

#options = apriltag.DetectorOptions(families="tag36h11")
detector = Detector(families="tag36h11")

#plt.ion()
#fig, ax = plt.subplots()
#ax.set_xlim([-2,2])
#ax.set_ylim([-2,2])

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
    #cv2.imshow('frame', frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break
    if __debug__:
        t2 = time.time()
    #print("Runtime: " + str(t2-t1))
    rot, trans, r_id = print_results(results)
    pose.append([t2, r_id, trans[0], trans[1], trans[2], rot[0], rot[1], rot[2]])

#save_to_csv(pose)

# When everything done, release the capture
cap.release() 
cv2.destroyAllWindows()
