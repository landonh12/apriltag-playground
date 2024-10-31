import cv2
import time
import argparse

# open video0
cap = cv2.VideoCapture(0)
time.sleep(1)
if(not cap.isOpened()):
    exit

# The control range can be viewed through v4l2-ctl -L
#cap.set(cv2.CAP_PROP_BRIGHTNESS, 64)
#cap.set(cv2.CAP_PROP_CONTRAST, 0)

# get settings
print(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
print(cap.get(cv2.CAP_PROP_EXPOSURE))

# Set exposure
print("Setting exposure...")
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, 50)

# Print exposure after setting
print(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE))
print(cap.get(cv2.CAP_PROP_EXPOSURE))

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

ret,frame = cap.read()

print(ret)

x1, y1, x2, y2 = (0, 0, 1920, int(1080/2))
      
frame = frame[y1:y2,x1:x2]

parser = argparse.ArgumentParser(description='capture images for camera calibration')
parser.add_argument('-n', '--num', metavar='N', type=int,
                        required=True,
                        help='number of image file')

options = parser.parse_args()

num = options.num

cv2.imwrite('/Users/landon/Documents/Personal/FRC/git/apriltag-playground/calib_images/%d.png' % num, frame)