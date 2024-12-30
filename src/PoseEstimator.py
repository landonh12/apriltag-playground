import numpy as np
import cv2
from pupil_apriltags import Detector
import time
import math
import subprocess
from TagPoseKalmanFilter import TagPoseKalmanFilter

class PoseEstimator:

    def __init__(self, exposure=1, res_width=1280, res_height=800, debug=False):
        # Camera Intrinsics
        # Arducam intrinsics
        self.camera_params = [941.287545323049, 936.9720277635304, 660.6219235101386, 413.5921687946573]
        #self.camera_params = (np.float64(674.3181908199018), np.float64(667.4179040814125), np.float64(1018.1982259356148), np.float64(248.77678435183148))
        self.detector = Detector(families="tag36h11", nthreads=4, quad_decimate=1)
        self.exposure = exposure
        self.res_width = 1280
        self.res_height = 800
        self.TAG_SIZE = 0.1651
        self.debug = debug
        self.cap = []
        self.cap.append(self.__setup_camera(0))
        self.cap.append(self.__setup_camera(1))
        self.pose_buffer = []
        self.kalman_filter = TagPoseKalmanFilter()
        self.previous_detections = []
        # Tag coordinates are indexed by ID + 1 (i.e. index 0 is tag 1).
        # Format is X Y Z Roll Pitch Yaw
        # Units are inches and degrees
        # Origin is source side corner, blue side of field
        # +x goes along long field wall
        # +y goes along blue driver station wall
        # +z is up from the floor. Floor is 0 z
        # Yaw rotation is right hand rule. 0 degrees is facing the red alliance driver station.
        # https://firstfrc.blob.core.windows.net/frc2024/FieldAssets/2024LayoutMarkingDiagram.pdf
        self.tag_coordinates = np.zeros((16,6))
        # 0.1651 meters = 6.5 inches
        self.tag_coordinates[2] = [17 + (1/8), 0, 16, 0, 0, 0]
        self.tag_coordinates[6] = [-17 - (1/8), 0, 16, 0, 0, 0]
        self.tag_coordinates[11] = [0, 0, 16, 0, 0, 0]

        ''' Field coordinates .. no longer have a field to use
        self.tag_coordinates = np.array([[593.68, 9.68, 53.38, 0, 0, 120],
                                        [637.21, 34.79, 53.38, 0, 0, 120],
                                        [652.73, 196.17, 57.13, 0, 0, 180],
                                        [652.73, 218.42, 57.13, 0, 0, 180],
                                        [578.77, 323.00, 53.38, 0, 0, 270],
                                        [72.5, 323.00, 53.38, 0, 0, 270],
                                        [-1.50, 218.42, 57.13, 0, 0, 0],
                                        [-1.50, 196.17, 57.13, 0, 0, 0],
                                        [14.02, 34.79, 53.38, 0, 0, 60],
                                        [57.54, 9.68, 53.38, 0, 0, 60],
                                        [468.69, 146.19, 52.00, 0, 0, 300],
                                        [468.69, 177.10, 52.00, 0, 0, 60],
                                        [441.74, 161.62, 52.00, 0, 0, 180],
                                        [209.48, 161.62, 52.00, 0, 0, 0],
                                        [182.73, 177.10, 52.00, 0, 0, 120],
                                        [182.73, 146.19, 52.00, 0, 0, 240]])
        '''

    # returns cap
    def __setup_camera(self, port=0):
        cap = cv2.VideoCapture(port)
        print(cap)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.res_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.res_height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        #self.ffmpeg_process = self.open_ffmpeg()
        return cap

    # returns tag results
    def __find_tags(self, camera):
        ret, frame = self.cap[camera].read()
        print(ret)
        """
        if(ret):
            if(side == 'rear'):
                x1, y1, x2, y2 = (0, 0, 1920, int(1080/2))
                frame = frame[y1:y2,x1:x2]
            if(side == 'front'):
                x1, y1, x2, y2 = (0, int(1080/2), 1920, 1080)
                frame = frame[y1:y2,x1:x2]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.detector.detect(gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size = self.TAG_SIZE)
        if self.debug == True:
            frame = self.draw_lines(frame, results)
            cv2.imshow('frame',frame)
        
        #self.ffmpeg_process.stdin.write(frame.astype(np.uint8).tobytes())
        #print(results)
        return results
    
    # This returns 1x3 vec in format roll pitch yaw (degrees)
    def __convert_rodrigues_to_rot(self, rot_vec):
        R, _ = cv2.Rodrigues(np.array(rot_vec))
        #print(R)
        return np.array([np.rad2deg(R[2][0]), np.rad2deg(R[0][0]), np.rad2deg(R[1][0])])
    
    def draw_lines(self, frame, results):
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

    # returns camera pose
    # format: "id"
    #         "trans_vec"
    #         "rot_vec"
    def get_camera_pose(self, camera):
        num_detections = 0
        results = self.__find_tags(camera)
        detections = np.zeros((16,6))

        if(len(results) == 0):
            self.kalman_filter.P *= 1.05  # Inflate covariance to increase uncertainty
            self.kalman_filter.predict()
            return self.kalman_filter.get_state()
        
        if(len(results) > 0):
            for i in range(len(results)):
                # Save ID
                #detections = results[i].tag_id
                id = results[i].tag_id - 1
                if id > 15:
                    break
                #print(id)
                num_detections = num_detections + 1

                print(results[i].pose_err)
                # Get Rotation and Translation Vectors
                R = results[i].pose_R
                cam_trans_vec = results[i].pose_t
                
                # If we get fudged numbers, do transpose before invert
                # Get trans and rot vectors
                cam_trans_vec = -R.transpose() @ cam_trans_vec
                rot_vec = self.__convert_rodrigues_to_rot(R)

                # print(cam_trans_vec)
                # print(rot_vec)

                # Camera pose relative to tag:
                # distance away: z (negative)
                # up/down: y (up is negative)
                # left/right: x (right is positive)

                # Field coordinates:
                # x is positive starting from blue driver station wall
                # y is positive starting from source side wall
                # z is positive going up

                # Do math here to get field coordinates using tag field coordinates
                # tag coordinates are in inches and degrees, make sure to convert to meters and degrees.

                tag_x = self.tag_coordinates[id][0]*0.0254
                tag_y = self.tag_coordinates[id][1]*0.0254
                tag_z = self.tag_coordinates[id][2]*0.0254
                tag_rot = self.tag_coordinates[id][5]

                cam_x = cam_trans_vec[0][0]
                cam_y = cam_trans_vec[1][0]
                cam_z = cam_trans_vec[2][0]

                trans_vec = np.zeros(3)
                trans_vec[0] = tag_x - (cam_z*math.cos(np.deg2rad(tag_rot))) - (cam_x*math.sin(np.deg2rad(tag_rot)))# x field pose
                trans_vec[1] = tag_y - (cam_z*math.sin(np.deg2rad(tag_rot))) + (cam_x*math.cos(np.deg2rad(tag_rot)))# y field pose
                trans_vec[2] = cam_y

                # trans_vec[0][0] = (self.tag_coordinates[id-1][0]*0.0254) + trans_vec[0][0] # x cam pose
                # trans_vec[1][0] = (self.tag_coordinates[id-1][1]*0.0254) + trans_vec[1][0] # y cam pose
                # trans_vec[2][0] = (self.tag_coordinates[id-1][2]*0.0254) + trans_vec[2][0] # z cam pose
                rot_vec[2] = self.tag_coordinates[id][5] + rot_vec[2]

                if(self.debug):
                    print(trans_vec)
                    print(rot_vec)

                #reshape trans and rot vecs? TODO
                detections[id] = [trans_vec[0], trans_vec[1], trans_vec[2],
                                    rot_vec[0], rot_vec[1], rot_vec[2]]
                real_id = id + 1
                print("id: %d" % real_id)
                for i in range(6):
                    print("%.3f" % detections[id][i])
                
                #z = np.array([trans_vec[0], trans_vec[1], trans_vec[2], rot_vec[0], rot_vec[1], rot_vec[2]])

                print("updating kalman filter")
                self.kalman_filter.update(detections[id])
        
        ''' Old averaging sensor fusion - remove
        fused_detections = np.zeros(6)
        for i in range(len(fused_detections)):
            for j in range(len(detections)):
                fused_detections[i] = fused_detections[i] + detections[j][i]
            fused_detections[i] = fused_detections[i] / num_detections
        '''     
        self.kalman_filter.predict()
        fused_detections = self.kalman_filter.get_state()

        return fused_detections


                







        