#!/usr/bin/env python3

# Python Libraries
import sys
import time

# NumPy
import numpy as np

# OpenCV
import cv2
from cv_bridge import CvBridge

# ROS Libraries
import rospy
import roslib

# ROS Message Types
from sensor_msgs.msg import CompressedImage

class LaneDetector:
    def _init_(self):
        self.cv_bridge = CvBridge()

        # Subscribing to the image topic
        self.image_sub = rospy.Subscriber('/akandb/camera_node/image/compressed', CompressedImage, self.image_callback, queue_size=1)

        rospy.init_node("lane_detector")

    def image_callback(self, msg):
        rospy.loginfo("Image received")

        # Convert compressed image to OpenCV format
        img = self.cv_bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        # Crop the image to focus on the road region (modify according to your camera's perspective)
        cropped_img = img[200:600, :]

        # Convert cropped image to HSV color space
        hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        # Define white color range in HSV
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([255, 25, 255])

        # Define yellow color range in HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Create masks for white and yellow colors
        white_mask = cv2.inRange(hsv_img, lower_white, upper_white)
        yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)

        # Apply Canny edge detection
        edges = cv2.Canny(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY), 50, 150)

        # Apply Hough Transform to detect lines
        white_lines = cv2.HoughLinesP(white_mask, 1, np.pi/180, 50, minLineLength=30, maxLineGap=100)
        yellow_lines = cv2.HoughLinesP(yellow_mask, 1, np.pi/180, 50, minLineLength=30, maxLineGap=100)

        # Draw white lines
        if white_lines is not None:
            for line in white_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(cropped_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw yellow lines
        if yellow_lines is not None:
            for line in yellow_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(cropped_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Display the processed image
        cv2.imshow('Lane Detection', cropped_img)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()

if _name_ == "_main_":
    try:
        lane_detector = LaneDetector()
        lane_detector.run()
    except rospy.ROSInterruptException:
        pass