#!/usr/bin/python

from ikpy.chain import Chain
import os
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import rospy
from tf.transformations import *
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped, PointStamped, Point, Quaternion
from visualization_msgs.msg import *
from sensor_msgs.msg import JointState
import rospkg
pkgPath = rospkg.RosPack()

from networks import RMDVAE
from nuitrack_node import NuitrackROS
from phd_utils.nuitrack import joints_idx


from matplotlib.pyplot import get_cmap
cmap = get_cmap('viridis')
cmap_idx = [0.5, 0.2, 0.9]

class RMDVAEHRINode:
	def callback(self, data):
		self.state_cb = data
		self.joint_values_left = np.array(self.state_cb.position)[9:16]
		self.joint_values_right = np.array(self.state_cb.position)[0:7]


	def __init__(self, ckpt_path):
		self.window_length=5
		input_dim=36*self.window_length
		self.readings = torch.zeros((input_dim,), device=device)
		ckpt = torch.load(ckpt_path)
		self.model = RMDVAE(input_dim,6*self.window_length,ckpt['args']).to(device)
		self.model.load_state_dict(ckpt['model'])
		self.model.eval()
			
		self.state = JointState()
		self.state.name = ['panda_right_joint1', 'panda_right_joint2', 'panda_right_joint3', 'panda_right_joint4',
						   'panda_right_joint5', 'panda_right_joint6', 'panda_right_joint7',
						   'panda_right_finger_joint1', 'panda_right_finger_joint2', 'panda_left_joint1',
						   'panda_left_joint2', 'panda_left_joint3', 'panda_left_joint4', 'panda_left_joint5',
						   'panda_left_joint6', 'panda_left_joint7', 'panda_left_finger_joint1',
						   'panda_left_finger_joint2']
		self.state_cb = JointState()
		self.sub = rospy.Subscriber('/panda_dual/joint_states', JointState, self.callback)
		self.pub = rospy.Publisher('/panda_dual/multi_mode_controller/desired_joint_position', JointState,
								   queue_size=10)
		
		self.joint_values_left = np.zeros(7, dtype='float')
		self.joint_values_right = np.zeros(7, dtype='float')
		self.panda_left_arm_chain = Chain.from_json_file(os.path.join(pkgPath.get_path('rmdn_hri_ros'),"robots/panda_left_arm.json"))
		self.panda_right_arm_chain = Chain.from_json_file(os.path.join(pkgPath.get_path('rmdn_hri_ros'),"robots/panda_right_arm.json"))
		

		# self.tfBuffer = tf2_ros.Buffer()
		# self.listener = tf2_ros.TransformListener(self.tfBuffer)
		# try:
		# 	self.baselink2leftlink0 = self.tfBuffer.lookup_transform('panda_left_link0', 'base_link', rospy.Time(0), rospy.Duration(1.0))
		# except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
		# 	rospy.loginfo('No transform between base_link and panda_left_link0')
		# 	rospy.signal_shutdown('No transform between base_link and panda_left_link0')
		# 	return

		# self.goal_pub = rospy.Publisher('/panda_dual/dual_arm_cartesian_impedance_controller/centering_frame_target_pose', PoseStamped, queue_size=10)
		# self.goal_msg = PoseStamped()
		# self.goal_msg.header.frame_id = 'panda_left_link0'
		# self.goal_msg.pose.orientation.x = 0.4618
		# self.goal_msg.pose.orientation.y = -0.7329
		# self.goal_msg.pose.orientation.z = 0.1907
		# self.goal_msg.pose.orientation.w = 0.4618

		self.robot_endeff_msg = MarkerArray()
		self.markerarray_pub = rospy.Publisher('/visualization_marker_array_hands', MarkerArray, queue_size=10)
		
		self.started = False
		self.hidden = None

		self.nuitrack = NuitrackROS(height=480, width=848, horizontal=True)
		self.history = []
		self.stamp = None


	def go_to_position_ik(self, l, r, loop=False):
		frame_target_left = np.array([[1.,0,0,0], [0,0,-1,0.], [0,1.,0,0], [0,0,0,1.]])
		frame_target_left[:3, 3] = l
		ik_left = self.panda_left_arm_chain.inverse_kinematics_frame(frame_target_left,initial_position=self.panda_left_arm_chain.active_to_full(np.array(self.state_cb.position)[9:16], initial_position=[0] * len(self.panda_left_arm_chain.links)), optimizer='least_squares', orientation_mode='all')
		iks_left = self.panda_left_arm_chain.active_from_full(ik_left)
		left = np.array(iks_left)
		frame_target_right = np.array([[1.,0,0,0], [0,0,1,0.], [0,-1.,0,0], [0,0,0,1.]])
		frame_target_right[:3, 3] = r
		ik_right = self.panda_right_arm_chain.inverse_kinematics_frame(frame_target_right,initial_position=self.panda_right_arm_chain.active_to_full(np.array(self.state_cb.position)[0:7], initial_position=[0] * len(self.panda_right_arm_chain.links)), optimizer='least_squares', orientation_mode='all')
		iks_right = self.panda_right_arm_chain.active_from_full(ik_right)

		right = np.array(iks_right)

		# self.state.position = [right[0], right[1], right[2], right[3], right[4], right[5], right[6],self.state_cb.position[7], self.state_cb.position[8], left[0], left[1], left[2], left[3], left[4], left[5], left[6], self.state_cb.position[16], self.state_cb.position[17]]
		# self.pub.publish(self.state)
		if loop:
			i = 0.1
			# for i in np.linspace(0.0, 1.0, num=20):
			while not ((np.linalg.norm(left - self.joint_values_left) < 0.05) & (
						np.linalg.norm(right - self.joint_values_right) < 0.05)):
				self.go_to_joint_position(i * left + (1.0 - i) * self.joint_values_left,
									i * right + (1.0 - i) * self.joint_values_right)
				i += 0.05
				i = min(i,1)
				rospy.Rate(30).sleep()
				if rospy.is_shutdown():
					break
		else:
			i = 0.5
			self.go_to_joint_position(i * left + (1.0 - i) * self.joint_values_left,
									i * right + (1.0 - i) * self.joint_values_right)

	def go_to_joint_position(self, left, right):
		self.state.position = [right[0], right[1], right[2], right[3], right[4], right[5], right[6], self.state_cb.position[7], self.state_cb.position[8], left[0], left[1], left[2], left[3], left[4], left[5], left[6], self.state_cb.position[16], self.state_cb.position[17]]
		self.pub.publish(self.state)

	def make_marker(self, x, y, z, r, g, b, stamp):
		marker = Marker()
		marker.ns = "rmdvae_gen"
		marker.header.frame_id = 'base_link'
		marker.id = len(self.robot_endeff_msg.markers)
		marker.lifetime = rospy.Duration(0)
		marker.frame_locked = False
		marker.action = Marker.ADD

		marker.type = Marker.SPHERE
		
		marker.color.r = r
		marker.color.g = g
		marker.color.b = b
		marker.color.a = 0.2
		
		marker.scale.x = marker.scale.y = marker.scale.z = 0.075
		
		marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)
		marker.pose.position.x = x
		marker.pose.position.y = y
		marker.pose.position.z = z

		marker.header.stamp = stamp

		self.robot_endeff_msg.markers.append(marker)

	def observe_human_optitrack(self):
		try:
			# human_right_transfom = self.tfBuffer.lookup_transform('base_link', 'right', rospy.Time(0))
			human_left_transfom = self.tfBuffer.lookup_transform('base_link', 'left', rospy.Time(0))
			# robot_right_transfom = self.tfBuffer.lookup_transform('base_link', 'panda_right_link8', rospy.Time(0))
			# robot_left_transfom = self.tfBuffer.lookup_transform('/base_link', '/panda_left_link8', rospy.Time(0))
		except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
			return

		stamp = rospy.Time.now()
		# self.make_marker(human_right_transfom.transform.translation.x, human_right_transfom.transform.translation.y, human_right_transfom.transform.translation.z, 1, 0, 0, stamp)
		self.make_marker(human_left_transfom.transform.translation.x, human_left_transfom.transform.translation.y, human_left_transfom.transform.translation.z, 1, 0, 0, stamp)
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

	def step_optitrack(self):
		if not self.started:
			return

		h_mean, h_std, h_alpha, r_out_h, r_out_components, self.hidden = self.model.forward_step(self.readings[None], self.hidden)
		robot_right_goal = r_out_h[0, :3].cpu().numpy()
		robot_left_goal = r_out_h[0, 3:].cpu().numpy()
		cartesian_goal = (robot_right_goal + robot_left_goal)*0.5
		
		self.goal_msg.pose.position = tf2_geometry_msgs.do_transform_point(
				PointStamped(point=Point(x = cartesian_goal[0], y = cartesian_goal[1], z = cartesian_goal[2])), 
				self.baselink2leftlink0
			).point
		
		stamp = rospy.Time.now()

		self.make_marker(robot_right_goal[0], robot_right_goal[1], robot_right_goal[2], 0, 0, 1, stamp)
		self.make_marker(robot_left_goal[0], robot_left_goal[1], robot_left_goal[2], 0, 0, 1, stamp)
		for i in range(3):
			self.make_marker(r_out_components[0, i, 0], r_out_components[0, i, 1], r_out_components[0, i, 2], cmap(cmap_idx[i])[0], cmap(cmap_idx[i])[1], cmap(cmap_idx[i])[2], stamp)
			self.make_marker(r_out_components[0, i, 3], r_out_components[0, i, 4], r_out_components[0, i, 5], cmap(cmap_idx[i])[0], cmap(cmap_idx[i])[1], cmap(cmap_idx[i])[2], stamp)
			
		self.goal_msg.header.stamp = stamp
		self.goal_pub.publish(self.goal_msg)
		self.markerarray_pub.publish(self.robot_endeff_msg)


	def observe_human(self):
		_, skeleton, self.stamp = self.nuitrack.update()
		if len(skeleton)==0:
			return
		lhand_idx = joints_idx["left_wrist"] - 1
		rhand_idx = joints_idx["right_wrist"] - 1

		skeleton = (self.nuitrack.base2cam[:3,:3].dot(skeleton.T) + self.nuitrack.base2cam[:3,3:4]).T

		rhand = skeleton[rhand_idx]
		lhand = skeleton[lhand_idx]
		self.make_marker(rhand[0], rhand[1], rhand[2], 1, 0, 0, self.stamp)
		self.make_marker(lhand[0], lhand[1], lhand[2], 1, 0, 0, self.stamp)

		idx_list = np.array([joints_idx[i] for i in ["left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"]]) - 1

		current_readings = torch.Tensor(skeleton[idx_list]).to(device)


		if self.history == []:
			print('first')
			x_pos = current_readings.flatten()[None]
			self.history = torch.cat([x_pos, torch.zeros_like(x_pos)], dim=-1)
		if not self.started and ((self.history[-1, [6,7,8,15,16,17]] - current_readings.flatten()[[6,7,8,15,16,17]])**2).sum() < 0.0005:
			print('Not yet started. Current displacement:', ((self.history[-1, [6,7,8,15,16,17]] - current_readings.flatten()[[6,7,8,15,16,17]])**2).sum())
			return
		
		x_pos = current_readings.flatten()[None]
		x_vel = x_pos - self.history[-1, :18]
		self.history = torch.vstack([self.history, torch.cat([x_pos, x_vel], dim=-1)])

		if self.history.shape[0] < self.window_length:
			return
		if not self.started:
			print('Starting',((self.history[-1, [6,7,8,15,16,17]] - current_readings.flatten()[[6,7,8,15,16,17]])**2).sum())
		self.started = True
		self.history = self.history[-self.window_length:]

	def step(self):
		if not self.started:
			return

		h_mean, h_std, h_alpha, r_out_h, r_out_components, self.hidden = self.model.forward_step(self.history.flatten()[None], self.hidden)
		robot_left_goal = r_out_h[0, :3].cpu().numpy()
		robot_right_goal = r_out_h[0, 3:6].cpu().numpy()
		
		self.make_marker(robot_right_goal[0], robot_right_goal[1], robot_right_goal[2], 0, 0, 1, self.stamp)
		self.make_marker(robot_left_goal[0], robot_left_goal[1], robot_left_goal[2], 0, 0, 1, self.stamp)
		for i in range(3):
			self.make_marker(r_out_components[0, i, 0], r_out_components[0, i, 1], r_out_components[0, i, 2], cmap(cmap_idx[i])[0], cmap(cmap_idx[i])[1], cmap(cmap_idx[i])[2], self.stamp)
			self.make_marker(r_out_components[0, i, 3], r_out_components[0, i, 4], r_out_components[0, i, 5], cmap(cmap_idx[i])[0], cmap(cmap_idx[i])[1], cmap(cmap_idx[i])[2], self.stamp)
			
		# self.goal_msg.header.stamp = self.stamp
		# self.goal_pub.publish(self.goal_msg)
		robot_left_goal[1] -= 0.1
		robot_right_goal[1] += 0.1
		self.go_to_position_ik(robot_left_goal, robot_right_goal)
		self.markerarray_pub.publish(self.robot_endeff_msg)
	

if __name__=='__main__':
	with torch.no_grad():
		rospy.init_node('rmdn_hri_node')
		rate = rospy.Rate(30)
		print('creating Controller')
		controller = RMDVAEHRINode(os.path.join(pkgPath.get_path('rmdn_hri_ros'),'models_final/rmdvae_nuitrack_kobo.pth'))
		controller.observe_human()
		rate.sleep()
		controller.observe_human()
		rate.sleep()
		count = 0
		hand_pos = []
		print('Starting Loop',rospy.is_shutdown())
		rate.sleep()
		while not rospy.is_shutdown():
			controller.observe_human()
			# if torch.all(controller.history.shape==torch.zeros_like(controller.readings)):
			if len(controller.history)==0:
				rate.sleep()
				continue

			count += 1
			if count<70:
				hand_pos.append(controller.history[-1, 15:18].cpu().numpy())
				# print(count, hand_pos[-1])
				rate.sleep()
				continue
			elif count == 70:
				hand_pos = np.mean(hand_pos[1:], 0)
				print('Calibration done', hand_pos)
			# controller.markerarray_pub.publish(controller.robot_endeff_msg)
			if controller.started:
			# if True:

				controller.step()
				# print(count, controller.readings[:6].cpu().numpy(), ((controller.readings[:6].cpu().numpy() - hand_pos)**2).sum())
				if count > 200 and ((controller.history[-1, 15:18].cpu().numpy() - hand_pos)**2).sum() < 0.05:
					# controller.goal_msg.pose.position.x = 0.336
					# controller.goal_msg.pose.position.y = -0.349
					# controller.goal_msg.pose.position.z = 0.121
					# controller.goal_msg.header.stamp = rospy.Time.now()
					# controller.goal_pub.publish(controller.goal_msg)
					# rospy.Rate(1).sleep()
					# controller.goal_msg.header.stamp = rospy.Time.now()
					# controller.goal_pub.publish(controller.goal_msg)
					controller.go_to_position_ik(np.array([0.55, 0.1175, 0.4]), np.array([0.55, -0.1175, 0.4]), loop=True)
					rospy.Rate(1).sleep()
					print('shutting down', count, ((controller.history[-1, 15:18].cpu().numpy() - hand_pos)**2).sum())
					rospy.signal_shutdown('Done')

			rate.sleep()