#!/usr/bin/env python
# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge, CvBridgeError
from opencv_object_tracking.msg import position_publish as mouthCenter
from std_msgs.msg import Int32
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sys
import rospy
import os
rospy.init_node('mouth_detector', anonymous=True)

# rospy.loginfo(os.path.realpath(__file__))
# rospy.loginfo(os.path.abspath(__file__))

# rospy.loginfo(os.getcwd())
rospy.loginfo(os.path.dirname(os.path.realpath(__file__)) )
#from opencv_object_tracking.msg import pixc
def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinate
# 	help="index of webcam on system")s
	A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
	B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar

# construct the argument parse and parse the argumentscdd

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=False, default='shape_predictor_68_face_landmarks.dat',
# 	help="path to facial landmark predictor")
# ap.add_argument("-w", "--webcam", type=int, default=0,
# 	help="index of webcam on system")
# args = vars(ap.parse_args())

# define one constants, for mouth aspect ratio to indicate open mouth
MOUTH_AR_THRESH = 0.79

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.dirname(os.path.realpath(__file__)) + '/shape_predictor_68_face_landmarks.dat')
#predictor = dlib.shape_predictor('/home/hyzn/catkin_ws/src/mouth_detect/scripts/shape_predictor_68_face_landmarks.dat')

#predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

# start the video stream thread
#print("[INFO] starting video stream thread...")
#vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

frame_width = 640
frame_height = 360

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
time.sleep(1.0)


# loop over frames from the video stream

##########image converter###########
class image_converter:
  def __init__(self):
    self.pcl2_matched = PointCloud2() #to match time with image center
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callbackImg_raw)
    self.pointCloud_sub = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.callback_pcl2)
    self.pointCloud_pub 	= rospy.Publisher("/mouthPos/pointCloud2"	,PointCloud2, queue_size=1) #from sensor_msgs.msg import PointCloud2 as pcl2
    self.mouthCenter_pub 	= rospy.Publisher("/mouthPos/mouthCenter"	,mouthCenter, queue_size=1) #from opencv_object_tracking.msg import position_publish as mouthCenter
    self.image_pub 			= rospy.Publisher("/mouthPos/raw_image"		,Image		, queue_size=1)
  def callback_pcl2(self, data) : 
	  self.pcl2_matched = data
  def callbackImg_raw(self,data):
    try:
	frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
	frame = imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	msg_pcl2 = self.pcl2_matched

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the mouth coordinates, then use the
		# coordinates to compute the mouth aspect ratio
		mouth = shape[mStart:mEnd]

		mouthMAR = mouth_aspect_ratio(mouth)
		mar = mouthMAR
		# compute the convex hull for the mouth, then
		# visualize the mouth
		mouthHull = cv2.convexHull(mouth)
		msg_mouth = mouthCenter()
		

		
		msg_mouth.upper_pixel_x = shape[67, 0]
		msg_mouth.upper_pixel_y = shape[67, 1]
		msg_mouth.lower_pixel_x = shape[63, 0]
		msg_mouth.lower_pixel_y = shape[63, 1]
		
		msg_mouth.counter = 1
		
		#mouth_center = (px, py)
		
		#cv2.putText(frame, "MAR: {:.2f}".format(mar), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
		cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		self.mouthCenter_pub.publish(msg_mouth)
        # Draw text if mouth is open
		if mar > MOUTH_AR_THRESH:
			cv2.putText(frame, "Mouth is Open!", (30,60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
	# Write the frame into the file 'output.avi'
	#out.write(frame)
	# show the frame
	#cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	#if key == ord("q"):
	#	break
    except CvBridgeError as e:
      print(e)
#    (rows,cols,channels) = cv_image.shape
#    if cols > 60 and rows > 60 :
#      cv2.circle(cv_image, (50,50), 10, 255)

#    cv2.imshow("Image window", cv_image)
#    cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
      self.pointCloud_pub.publish(msg_pcl2)
    except CvBridgeError as e:
      print(e)

#	rospy.loginfo("pcl2")

###################################
def main(args):
  ic = image_converter()
  
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
# do a bit of cleanup
#cv2.destroyAllWindows()
#vs.stop()
