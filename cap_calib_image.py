import cv2
import time

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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

ret,frame = cap.read()

cv2.imwrite('/home/landon/personal/misc/calib_images/12.png', frame)