#!/usr/bin/python
import os
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from rmdn_hri.networks import RMDN

import rospy
import tf2_ros
import tf2_geometry_msgs
import rospkg
pkgPath = rospkg.RosPack()

from geometry_msgs.msg import PoseStamped, PointStamped, Point, PoseArray, Pose
from moveit_msgs.msg import DisplayRobotState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from pepper_controller_server.srv import JointTarget

from nuitrack_node import NuitrackROS
from utils import *

default_arm_joints = [1.5708, -0.109, 0.7854, 0.009, 1., 0.] # default standing angle values

class RMDNHRINode:
	def __init__(self, ckpt_path):
		super().__init__()
		input_dim=18
		ckpt = torch.load(ckpt_path)
		self.model = RMDN(input_dim,4,ckpt['args']).to(device)
		self.model.load_state_dict(ckpt['model'])
		self.model.eval()

		self.started = False
		self.hidden = None

		rospy.wait_for_service('/pepper_dcm/RightArm_controller/goal')
		self.send_target = rospy.ServiceProxy('/pepper_dcm/RightArm_controller/goal', JointTarget)
		self.state_pub = rospy.Publisher("display_robot_state", DisplayRobotState, queue_size=5)
		self.broadcaster = tf2_ros.StaticTransformBroadcaster()
		self.tfBuffer = tf2_ros.Buffer()
		self.listener = tf2_ros.TransformListener(self.tfBuffer)
		self.robot_hand_joint = 0.

		self.joint_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", 'RHand']
		self.state_msg = DisplayRobotState()
		self.state_msg.state.joint_state.header.frame_id = "base_footprint"
		self.state_msg.state.joint_state.name = self.joint_names
		self.joint_trajectory = JointTrajectory()
		self.joint_trajectory.header.frame_id = "base_footprint"
		self.joint_trajectory.joint_names = self.joint_names
		for i in range(self.model.num_components+1):
			self.joint_trajectory.points.append(JointTrajectoryPoint())
		self.joint_trajectory.points[0].effort = np.ones(len(self.joint_names)).tolist()
		self.joint_trajectory.points[0].effort[0] = 1.0
		self.joint_trajectory.points[0].positions = default_arm_joints

		self.nuitrack = NuitrackROS(height=480, width=848, horizontal=False)

		self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)

		self.prev_obs = None

	def joint_state_cb(self, msg:JointState):
		if len(msg.name)<=6:
			return
		self.joint_readings = np.array(msg.position[11:17])

		self.state_msg.state.joint_state = msg
		self.state_msg.state.joint_state.header.frame_id = "base_footprint"
		self.state_msg.state.joint_state.position = list(self.state_msg.state.joint_state.position)
		self.state_msg.state.joint_state.position[11:17] = list(self.joint_trajectory.points[0].positions)

	def observe_human(self):
		img, nui_skeleton, stamp = self.nuitrack.update()
		if img is None or len(nui_skeleton)==0:
			return [], None, stamp

		if img is None or len(nui_skeleton)==0:
			return [], None, stamp

		hand_pose = self.nuitrack.base2cam[:3,:3].dot(nui_skeleton[-1, :]) + self.nuitrack.base2cam[:3,3]

		return nui_skeleton, hand_pose, stamp

	def in_baselink(self, hand_pose):
		# Need pose in base_link frame for IK
		link_TF = self.tfBuffer.lookup_transform('base_link', 'base_footprint', rospy.Time(0))
		link_transform = ROS2mat(link_TF.transform)
		hand_pose = link_transform[:3,:3].dot(hand_pose) + link_transform[:3,3]
		return hand_pose - self.offset

	def publish(self, stamp):
		self.state_msg.state.joint_state.header.stamp = self.joint_trajectory.header.stamp = stamp
		self.send_target(self.joint_trajectory)
		self.state_pub.publish(self.state_msg)

	def step(self, nui_skeleton):
		if self.prev_obs is None:
			self.preproc_transformation = rotation_normalization(nui_skeleton)

		nui_skeleton = torch.Tensor(self.preproc_transformation[:3,:3].dot(nui_skeleton.T).T + self.preproc_transformation[:3,3]).to(device)

		if self.prev_obs is None:
			curr_pos = nui_skeleton[[-4,-3,-2], :].flatten()[None]
			self.prev_obs = curr_pos.clone()

		if not self.started and ((nui_skeleton[-2,:] - self.prev_obs[0, -3:])**2).sum() < 0.0005:
			print('Not yet started. Current displacement:', ((nui_skeleton[-2,:] - self.prev_obs[0, -3:])**2).sum())
			return

		curr_pos = nui_skeleton[[-4,-3,-2], :].flatten()[None]
		curr_vel = curr_pos - self.prev_obs
		self.prev_obs = curr_pos.clone()

		if not self.started:
			print('Starting',((nui_skeleton[-2,:] - self.prev_obs[0, -3:])**2).sum())
		self.started = True

		with torch.no_grad():
			h_mean, h_std, h_alpha, self.hidden = self.model.forward_step(torch.hstack([curr_pos, curr_vel]), self.hidden)
		h_mean = h_mean.cpu().numpy()
		h_alpha = h_alpha.cpu().numpy()
		for i in range(1, self.model.num_components+1):
			self.joint_trajectory.points[0].positions = h_mean[0, i-1].tolist()
			self.joint_trajectory.points[0].effort = h_alpha[0, i-1].tolist()
		print(h_mean, h_alpha)
		h_mean = (h_mean*h_alpha[..., None]).sum(1)

		self.joint_trajectory.points[0].positions = 0.2*np.array(self.joint_trajectory.points[0].positions) + 0.8*np.array(h_mean[0].tolist() + [1., 0.])
		self.joint_trajectory.points[0].positions[0] -= np.deg2rad(15)

		# Some IK based on the distance of the robot hand to the human hand

if __name__=='__main__':
	rospy.init_node('rmdn_hri_node')
	rate = rospy.Rate(100)
	print('creating Controller')
	controller = RMDNHRINode(os.path.join(pkgPath.get_path('rmdn_hri_ros'),'models/pepper_unimanual.pth'))
	controller.observe_human()
	count = 0
	hand_pos_init = []
	rospy.Rate(0.5).sleep()
	controller.joint_trajectory.points[0].effort[0] = 1.0
	rate.sleep()
	started = False
	while not rospy.is_shutdown():
		nui_skeleton, hand_pose, stamp = controller.observe_human()
		if len(nui_skeleton)!=0:
			count += 1
		if count < 20:
			if hand_pose is not None:
				hand_pos_init.append(hand_pose)
			controller.publish(stamp)
			rate.sleep()
			continue
		elif count == 20:
			hand_pos_init = np.mean(hand_pos_init, 0)
			print('Calibration ready')
		if not started and ((hand_pose - hand_pos_init)**2).sum() < 0.001:
			print('Not yet started. Current displacement:', ((hand_pose - hand_pos_init)**2).sum())
			continue
		else:
			started = True
		
		controller.step(nui_skeleton)
		controller.publish(stamp)
		rate.sleep()
		if started and count>100 and ((hand_pose - hand_pos_init)**2).sum() < 0.005: # hand_pose[2] < 0.63 and hand_pose[2] - prev_z < -0.005:
			break

	controller.joint_trajectory.points[0].effort[0] = 1.0
	controller.joint_trajectory.points[0].positions = default_arm_joints
	controller.publish(rospy.Time.now())
	rospy.Rate(1).sleep()
	rospy.signal_shutdown('done')