#!/usr/bin/python

from PyNuitrack import py_nuitrack

import cv2
import numpy as np

from cv_bridge import CvBridge
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
from tf.transformations import *
import rospy

from utils import *

class NuitrackWrapper:
	def __init__(self, height=480, width=848, horizontal=False):
		self._height = height
		self._width = width
		self._horizontal = horizontal

		self.init_nuitrack()
		print("Nuitrack Version:", self.nuitrack.get_version())
		print("Nuitrack License:", self.nuitrack.get_license())
	
	def init_nuitrack(self):
		self.nuitrack = py_nuitrack.Nuitrack()
		self.nuitrack.init()

		self.nuitrack.set_config_value("DepthProvider.Depth2ColorRegistration", "true")
		if not self._horizontal:
			self.nuitrack.set_config_value("DepthProvider.RotateAngle", "270")

		# Realsense Depth Module - force to 848x480 @ 30 FPS
		self.nuitrack.set_config_value("Realsense2Module.Depth.Preset", "5")
		self.nuitrack.set_config_value("Realsense2Module.Depth.RawWidth", str(self._width))
		self.nuitrack.set_config_value("Realsense2Module.Depth.RawHeight", str(self._height))
		self.nuitrack.set_config_value("Realsense2Module.Depth.ProcessWidth", str(self._width))
		self.nuitrack.set_config_value("Realsense2Module.Depth.ProcessHeight", str(self._height))
		self.nuitrack.set_config_value("Realsense2Module.Depth.FPS", "30")

		# Realsense RGB Module - force to 848x480 @ 30 FPS
		self.nuitrack.set_config_value("Realsense2Module.RGB.RawWidth", str(self._width))
		self.nuitrack.set_config_value("Realsense2Module.RGB.RawHeight", str(self._height))
		self.nuitrack.set_config_value("Realsense2Module.RGB.ProcessWidth", str(self._width))
		self.nuitrack.set_config_value("Realsense2Module.RGB.ProcessHeight", str(self._height))
		self.nuitrack.set_config_value("Realsense2Module.RGB.FPS", "30")

		devices = self.nuitrack.get_device_list()
		for i, dev in enumerate(devices):
			print(dev.get_name(), dev.get_serial_number())
			if i == 0:
				#dev.activate("ACTIVATION_KEY") #you can activate device using python api
				print(dev.get_activation())
				self.nuitrack.set_device(dev)

		self.nuitrack.create_modules()
		self.nuitrack.run()

	def reset_nuitrack(self):
		try:
			self.nuitrack.release()
		except:
			print("Could not release Nuitrack, just resetting it")

		self.init_nuitrack()

	def update(self):
		self.nuitrack.update()
		
		self._depth_img = self.nuitrack.get_depth_data()
		self._color_img = self.nuitrack.get_color_data()
		if not self._depth_img.size or not self._color_img.size:
			return None, []
		display_img = self._color_img.copy()

		data = self.nuitrack.get_skeleton()
		if len(data.skeletons)==0:
			return display_img, []

		skeleton = np.zeros((14,3))
		for bone in connections:
			j0 = data.skeletons[0][joints_idx[bone[0]]]
			j1 = data.skeletons[0][joints_idx[bone[1]]]
			x0 = (round(j0.projection[0]), round(j0.projection[1]))
			x1 = (round(j1.projection[0]), round(j1.projection[1]))
			cv2.line(display_img, x0, x1, line_color, 5)
		for i in range(1,15):
			x = (round(data.skeletons[0][i].projection[0]), round(data.skeletons[0][i].projection[1]))
			cv2.circle(display_img, x, 15, point_color, -1)
			skeleton[i-1] = data.skeletons[0][i].real * 0.001
		
		skeleton[:, 1] *= -1
		return display_img, skeleton

	def __del__(self):
		self.nuitrack.release()

class NuitrackROS(NuitrackWrapper):
	def __init__(self, height=480, width=848, camera_link='camera_link', horizontal=False):
		# Ideally use a tf listener, but this is easier
		xyz = [rospy.get_param("/tf_dynreconf_node/x", 0), rospy.get_param("/tf_dynreconf_node/y", 0), rospy.get_param("/tf_dynreconf_node/z", 0)]
		rpy = [rospy.get_param("/tf_dynreconf_node/roll", 0), rospy.get_param("/tf_dynreconf_node/pitch", 0), rospy.get_param("/tf_dynreconf_node/yaw", 0)]
		self.base2cam = euler_matrix(*rpy)
		self.base2cam[:3, 3] = np.array(xyz)
		super().__init__(height, width, horizontal)
		
		intrinsics = intrinsics_horizontal if horizontal else intrinsics_vertical
		K = intrinsics[(self._width, self._height)]

		self._color_pub = rospy.Publisher('image_color', Image, queue_size=10)
		self._depth_pub = rospy.Publisher('image_depth', Image, queue_size=10)
		self._display_pub = rospy.Publisher('image_display', Image, queue_size=10)
		self._camerainfo_pub = rospy.Publisher('camera_info', CameraInfo, queue_size=10)
		self._viz_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)

		self._camerainfo = CameraInfo(
									height=height,
									width = width,
									distortion_model = "plumb_bob",
									D = np.zeros(5),
									K = K.flatten(),
									R = np.eye(3).flatten(),
									P = np.hstack([K,np.zeros(3)[:, None]]).flatten()
								)
		
		self._markerarray_msg = MarkerArray()
		lines = []
		for i in range(14):
			marker = Marker()
			line_strip = Marker()
			line_strip.ns = marker.ns = "nuitrack_skeleton"
			marker.id = i
			line_strip.id = i + 14
			line_strip.lifetime = marker.lifetime = rospy.Duration(0.5)
			line_strip.frame_locked = marker.frame_locked = False
			line_strip.action = marker.action = Marker.ADD

			marker.type = Marker.SPHERE
			line_strip.type = Marker.LINE_STRIP

			line_strip.color.a = line_strip.color.r = marker.color.a = marker.color.g = 1
			line_strip.color.g = line_strip.color.b = marker.color.b = marker.color.r = 0

			marker.scale.x = marker.scale.y = marker.scale.z = 0.05
			line_strip.scale.x = 0.02

			line_strip.pose.orientation = marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)

			line_strip.points = [Point(), Point()]

			self._markerarray_msg.markers.append(marker)
			lines.append(line_strip)
		self._markerarray_msg.markers = self._markerarray_msg.markers + lines[:-1]

		self._header = Header(frame_id = camera_link, seq = 0)
		
		self._bridge = CvBridge()

	def publish_img(self, publisher, image, encoding):
		if publisher.get_num_connections() > 0:
			msg = self._bridge.cv2_to_imgmsg(image, encoding=encoding)
			msg.header = self._header
			publisher.publish(msg)
			
	def update(self):
		display_img, skeleton = super().update()
		self._header.stamp = rospy.Time.now()
		if display_img is None:
			return None, [], self._header.stamp
		self._header.seq += 1

		self.publish_img(self._color_pub, self._color_img, "bgr8")
		self.publish_img(self._display_pub, display_img, "bgr8")
		self.publish_img(self._depth_pub, self._depth_img, "passthrough")

		if self._camerainfo_pub.get_num_connections() > 0:
			self._camerainfo.header = self._header
			self._camerainfo_pub.publish(self._camerainfo)
		
		if self._viz_pub.get_num_connections() > 0 and len(skeleton)>0:
			# for i in range(14):
			for i in [-4,-3,-2]:
				self._markerarray_msg.markers[i].pose = mat2Pose(skeleton[i])
				self._markerarray_msg.markers[i].header = self._header

			# for i in range(len(connections)):
			# 	bone = connections[i]
			# 	self._markerarray_msg.markers[i+14].points[0].x = skeleton[joints_idx[bone[0]]-1,0]
			# 	self._markerarray_msg.markers[i+14].points[0].y = skeleton[joints_idx[bone[0]]-1,1]
			# 	self._markerarray_msg.markers[i+14].points[0].z = skeleton[joints_idx[bone[0]]-1,2]
			# 	self._markerarray_msg.markers[i+14].points[1].x = skeleton[joints_idx[bone[1]]-1,0]
			# 	self._markerarray_msg.markers[i+14].points[1].y = skeleton[joints_idx[bone[1]]-1,1]
			# 	self._markerarray_msg.markers[i+14].points[1].z = skeleton[joints_idx[bone[1]]-1,2]
			# 	self._markerarray_msg.markers[i+14].header = self._header
			self._viz_pub.publish(self._markerarray_msg)

		return display_img, skeleton, self._header.stamp

if __name__=="__main__":
	rospy.init_node("nuitrack_node")
	nuitrack = NuitrackROS(width=848, height=480, horizontal=False)
	rate = rospy.Rate(500)
	
	t = TransformStamped()
	t.header.frame_id = 'base_footprint'
	t.child_frame_id = 'hand'
	broadcaster = tf2_ros.StaticTransformBroadcaster()
	
	while not rospy.is_shutdown():
		display_img, skeleton, stamp = nuitrack.update()
		if len(skeleton)>0:
			hand_pose = skeleton[-1, :]
			hand_pose = nuitrack.base2cam[:3,:3].dot(hand_pose) + nuitrack.base2cam[:3,3]
			t.transform = mat2TF(hand_pose)
			t.header.stamp = stamp
			broadcaster.sendTransform(t)
		rate.sleep()
		