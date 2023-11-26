#!/usr/bin/python
import os
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from ikpy.chain import Chain

from rmdn_hri.networks import RMDVAE

import rospy
import tf2_ros
import tf2_geometry_msgs
import rospkg
rospack = rospkg.RosPack()

from geometry_msgs.msg import PoseStamped, PointStamped, Point, PoseArray, Pose
from moveit_msgs.msg import DisplayRobotState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from pepper_controller_server.srv import JointTarget

from nuitrack_node import NuitrackROS
from utils import *

from matplotlib.pyplot import get_cmap
cmap = get_cmap('viridis')
cmap_idx = [0.5, 0.2, 0.9]

default_arm_joints = [1.5708, -0.109, 0.7854, 0.009, 1., 0.] # default standing angle values

class RMDVAEHRINode:
	def __init__(self, ckpt_path):
		super().__init__()
		input_dim=18*5
		ckpt = torch.load(ckpt_path)
		self.model = RMDVAE(input_dim,20,ckpt['args']).to(device)
		self.model.load_state_dict(ckpt['model'])
		self.model.eval()

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
		self.joint_trajectory.points.append(JointTrajectoryPoint())
		self.joint_trajectory.points[0].effort = np.ones(len(self.joint_names)).tolist()
		self.joint_trajectory.points[0].effort[0] = 1.0
		self.joint_trajectory.points[0].positions = default_arm_joints

		self.pepper_chain = Chain.from_json_file(os.path.join(rospack.get_path('mild_hri_ros'), "resources", "pepper", "pepper_right_arm.json"))

		self.nuitrack = NuitrackROS(height=480, width=848, horizontal=False)

		self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)

		self.prev_obs = None

		self.robot_endeff_msg = MarkerArray()
		self.markerarray_pub = rospy.Publisher('/visualization_marker_array', MarkerArray, queue_size=10)
		
		self.started = False
		self.hidden = None
		self.history = []


	def make_marker(self, x, y, z, r, g, b, stamp):
		marker = Marker()
		marker.ns = "rmdvae_gen"
		marker.header.frame_id = 'base_link'
		marker.id = len(self.robot_endeff_msg.markers)
		marker.lifetime = rospy.Duration(0.5)
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
		if self.started:
			hand_pose_baselink = self.in_baselink(hand_pose)
			self.make_marker(hand_pose_baselink[0], hand_pose_baselink[1], hand_pose_baselink[2], 1, 0, 0, stamp)
		
		return nui_skeleton, hand_pose, stamp

	def in_baselink(self, hand_pose):
		# Need pose in base_link frame for IK
		link_TF = self.tfBuffer.lookup_transform('base_link', 'base_footprint', rospy.Time(0))
		link_transform = ROS2mat(link_TF.transform)
		hand_pose = link_transform[:3,:3].dot(hand_pose) + link_transform[:3,3]
		return hand_pose

	def publish(self, stamp):
		self.state_msg.state.joint_state.header.stamp = self.joint_trajectory.header.stamp = stamp
		self.send_target(self.joint_trajectory)
		self.state_pub.publish(self.state_msg)
		self.markerarray_pub.publish(self.robot_endeff_msg)

	def step(self, nui_skeleton):
		if self.prev_obs is None:
			self.preproc_transformation = rotation_normalization(nui_skeleton)

		nui_skeleton = torch.Tensor(self.preproc_transformation[:3,:3].dot(nui_skeleton.T).T + self.preproc_transformation[:3,3]).to(device)
		if self.prev_obs is None:
			curr_pos = nui_skeleton[[-4,-3,-2], :].flatten()[None]
			self.prev_obs = curr_pos.clone()
			self.history = torch.cat([curr_pos, torch.zeros_like(curr_pos)], dim=-1)
		
		if not self.started and ((nui_skeleton[-2,:] - self.prev_obs[0, -3:])**2).sum() < 0.0005:
			print('Not yet started. Current displacement:', ((nui_skeleton[-2,:] - self.prev_obs[0, -3:])**2).sum())
			return
		
		curr_pos = nui_skeleton[[-4,-3,-2], :].flatten()[None]
		curr_vel = curr_pos - self.prev_obs
		self.prev_obs = curr_pos.clone()
		
		self.history = torch.vstack([self.history, torch.cat([curr_pos, curr_vel], dim=-1)])

		if self.history.shape[0] < 5:
			return
		if not self.started:
			print('Starting',((nui_skeleton[-2,:] - self.prev_obs[0, -3:])**2).sum())
		self.started = True
		self.history = self.history[-5:]

		with torch.no_grad():
			# h_mean, h_std, h_alpha, self.hidden = self.model.forward_step(torch.hstack([curr_pos, curr_vel]), self.hidden)
			h_mean, h_std, h_alpha, r_out_h, r_out_components, self.hidden = self.model.forward_step(self.history.flatten()[None], self.hidden)
		
		self.joint_trajectory.points[0].positions = 0.2*np.array(self.joint_trajectory.points[0].positions) + 0.8*np.array(r_out_h[0, :4].tolist() + [1., 0.])
		self.joint_trajectory.points[0].positions[0] -= np.deg2rad(15)

		endeff = self.pepper_chain.forward_kinematics(self.pepper_chain.active_to_full(self.joint_trajectory.points[0].positions[:4], [0] * len(self.pepper_chain.links)))
		self.make_marker(endeff[0,3], endeff[1,3], endeff[2,3], 0, 0, 1, stamp)
		for i in range(self.model.num_components):
			r_out_components[0, i, 0] -= np.deg2rad(15)
			endeff = self.pepper_chain.forward_kinematics(self.pepper_chain.active_to_full(r_out_components[0, i, :4].cpu().numpy(), [0] * len(self.pepper_chain.links)))
			self.make_marker(endeff[0,3], endeff[1,3], endeff[2,3], cmap(cmap_idx[i])[0], cmap(cmap_idx[i])[1], cmap(cmap_idx[i])[2], stamp)

		# Some IK based on the distance of the robot hand to the human hand

if __name__=='__main__':
	rospy.init_node('rmdn_hri_node')
	rate = rospy.Rate(100)
	print('creating Controller')
	controller = RMDVAEHRINode(os.path.join(rospack.get_path('rmdn_hri_ros'),'models_final/rmdvae_nuisi_pepper.pth'))
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