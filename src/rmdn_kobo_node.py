#!/usr/bin/python
import os
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from rmdn_hri.networks import RMDN

import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, PointStamped, Point, PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray
import rospkg
pkgPath = rospkg.RosPack()

class RMDNHRINode:
	def __init__(self, ckpt_path):
		input_dim=6
		self.readings = torch.zeros((input_dim,), device=device)
		ckpt = torch.load(ckpt_path)
		self.model = RMDN(input_dim,12,ckpt['args']).to(device)
		self.model.load_state_dict(ckpt['model'])
		self.model.eval()

		self.tfBuffer = tf2_ros.Buffer()
		self.listener = tf2_ros.TransformListener(self.tfBuffer)
		try:
			self.baselink2leftlink0 = self.tfBuffer.lookup_transform('panda_left_link0', 'base_link', rospy.Time(0), rospy.Duration(1.0))
			self.leftlink02baselink = self.tfBuffer.lookup_transform('base_link','panda_left_link0', rospy.Time(0), rospy.Duration(1.0))
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
			rospy.loginfo('No transform between base_link and panda_left_link0')
			rospy.signal_shutdown('No transform between base_link and panda_left_link0')
			return

		self.goal_pub = rospy.Publisher('/panda_dual/dual_arm_cartesian_impedance_controller/centering_frame_target_pose', PoseStamped, queue_size=10)
		self.poses_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)

		self.markerarray_msg = MarkerArray()
		for n in range(4):
			for i in range(self.model.num_components):
				marker = Marker()
				marker.ns = "rmdn_outputs"
				marker.header.frame_id = 'base_link'
				marker.id = i + n * 4
				marker.lifetime = rospy.Duration(0.5)
				marker.frame_locked = False
				marker.action = Marker.ADD

				marker.type = Marker.ARROW

				marker.color.g = n==0 or n==3
				marker.color.b = n==1
				marker.color.r = n==2 or n==3

				marker.pose.orientation.x = 0.6530890570696277
				marker.pose.orientation.y = -0.27110217265848563
				marker.pose.orientation.z = 0.2701055162914406
				marker.pose.orientation.w = 0.653491411147435
				
				marker.scale.y = marker.scale.z = 0.05
				marker.scale.x = 0.5

				self.markerarray_msg.markers.append(marker)


		self.goal_msg = PoseStamped()
		self.goal_msg.header.frame_id = 'panda_left_link0'
		self.goal_msg.pose.orientation.x = 0.4618
		self.goal_msg.pose.orientation.y = -0.7329
		self.goal_msg.pose.orientation.z = 0.1907
		self.goal_msg.pose.orientation.w = 0.4618

		self.started = False
		self.hidden = None

	def observe_human(self):
		try:
			# human_right_transfom = self.tfBuffer.lookup_transform('base_link', 'right', rospy.Time(0))
			human_left_transfom = self.tfBuffer.lookup_transform('base_link', 'left', rospy.Time(0))
			# robot_right_transfom = self.tfBuffer.lookup_transform('base_link', 'panda_right_link8', rospy.Time(0))
			# robot_left_transfom = self.tfBuffer.lookup_transform('/base_link', '/panda_left_link8', rospy.Time(0))
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
			return

		current_readings = torch.Tensor([
								# human_right_transfom.transform.translation.x, human_right_transfom.transform.translation.y, human_right_transfom.transform.translation.z,
								human_left_transfom.transform.translation.x, human_left_transfom.transform.translation.y, human_left_transfom.transform.translation.z,
								# human_left_transfom.transform.translation.x - robot_right_transfom.transform.translation.x, 
								# human_left_transfom.transform.translation.y - robot_right_transfom.transform.translation.y, 
								# human_left_transfom.transform.translation.z - robot_right_transfom.transform.translation.z,
		]).to(device)


		if torch.all(self.readings==torch.zeros_like(self.readings)):
			print('first')
			# self.readings[:3] = torch.Tensor([human_right_transfom.transform.translation.x, human_right_transfom.transform.translation.y, human_right_transfom.transform.translation.z]).to(device)
			self.readings[:3] = torch.Tensor([human_left_transfom.transform.translation.x, human_left_transfom.transform.translation.y, human_left_transfom.transform.translation.z]).to(device)
			# self.readings[6:9] = torch.Tensor([
			# 										human_left_transfom.transform.translation.x - robot_right_transfom.transform.translation.x, 
			# 										human_left_transfom.transform.translation.y - robot_right_transfom.transform.translation.y, 
			# 										human_left_transfom.transform.translation.z - robot_right_transfom.transform.translation.z
			# ]).to(device)
			return
		if not self.started and ((self.readings[:3] - current_readings[:3])**2).sum() < 0.0005:
			# print('Not yet started. Current displacement:', ((self.readings[:6] - current_readings[:6])**2).sum())
			return
		
		self.readings[3:6] = current_readings[:3] - self.readings[:3]
		# self.readings[9:12] = current_readings[3:6] - self.readings[3:6]
		# self.readings[9:12] = current_readings[3:6] - self.readings[6:9]
		self.readings[:3] = current_readings[:3]
		# self.readings[6:9] = current_readings[3:6]
		# self.readings[12:15] = current_readings[6:9]
		
		if not self.started:
			print('Starting', ((self.readings[:3] - current_readings[:3])**2).sum())
		self.started = True

			
	def step(self):
		if not self.started:
			return

		h_mean, h_std, h_alpha, self.hidden = self.model.forward_step(self.readings[None], self.hidden)
		h_mean = h_mean.cpu().numpy()
		h_alpha = h_alpha.cpu().numpy()
		stamp = rospy.Time.now()
		for n in range(4):
			for i in range(self.model.num_components):
				self.markerarray_msg.markers[i+n*self.model.num_components].pose.position.x = h_mean[0, i, n*3]
				self.markerarray_msg.markers[i+n*self.model.num_components].pose.position.y = h_mean[0, i, n*3+1]
				self.markerarray_msg.markers[i+n*self.model.num_components].pose.position.z = h_mean[0, i, n*3+2]
				self.markerarray_msg.markers[i+n*self.model.num_components].color.a = h_alpha[0,i]
				self.markerarray_msg.markers[i+n*self.model.num_components].header.stamp = stamp
		h_mean = (h_mean*h_alpha[..., None]).sum(1)
		left_rmg = h_mean[0, :3]
		right_rmg = h_mean[0, 3:6]
		left_goal = h_mean[0, 6:9]
		right_goal = h_mean[0, 9:]

		rmg_center = (left_rmg + right_rmg)*0.5
		goal_center = (left_goal + right_goal)*0.5

		dist = np.linalg.norm(rmg_center - goal_center)**2
		w = np.exp(-500*dist)
		# w = 1

		target_pose = rmg_center + w*(goal_center - rmg_center)
		# self.goal_msg.pose.position.x = target_pose[0]
		# self.goal_msg.pose.position.y = target_pose[1]
		# self.goal_msg.pose.position.z = target_pose[2]

		self.goal_msg.pose.position = tf2_geometry_msgs.do_transform_point(
				PointStamped(point=Point(x = target_pose[0], y = target_pose[1], z = target_pose[2])), 
				self.baselink2leftlink0
			).point

		self.goal_msg.header.stamp = stamp
		self.goal_pub.publish(self.goal_msg)
		self.poses_pub.publish(self.markerarray_msg)

if __name__=='__main__':
	with torch.no_grad():
		rospy.init_node('rmdn_hri_node')
		rate = rospy.Rate(100)
		print('creating Controller')
		controller = RMDNHRINode(os.path.join(pkgPath.get_path('rmdn_hri_ros'),'models/rmdn_rmg_goal_trial0_400.pth'))

		count = 0
		hand_pos = []
		print('Starting Loop',rospy.is_shutdown())
		rate.sleep()
		while not rospy.is_shutdown():
			controller.observe_human()
			if torch.all(controller.readings==torch.zeros_like(controller.readings)):
				rate.sleep()
				continue

			count += 1
			if count<70:
				hand_pos.append(controller.readings[:6].cpu().numpy())
				# print(count, hand_pos[-1])
				rate.sleep()
				continue
			elif count == 70:
				hand_pos = np.mean(hand_pos[1:], 0)
				print('Calibration done', hand_pos)
			if controller.started:
			# if True:
				controller.step()
				# print(count, controller.readings[:6].cpu().numpy(), ((controller.readings[:6].cpu().numpy() - hand_pos)**2).sum())
				if count > 500 and ((controller.readings[:6].cpu().numpy() - hand_pos)**2).sum() < 0.03:
					controller.goal_msg.pose.position.x = 0.336
					controller.goal_msg.pose.position.y = -0.349
					controller.goal_msg.pose.position.z = 0.121
					controller.goal_msg.header.stamp = rospy.Time.now()
					controller.goal_pub.publish(controller.goal_msg)
					rospy.Rate(1).sleep()
					controller.goal_msg.header.stamp = rospy.Time.now()
					controller.goal_pub.publish(controller.goal_msg)
					rospy.Rate(1).sleep()
					rospy.signal_shutdown('Done')

			rate.sleep()