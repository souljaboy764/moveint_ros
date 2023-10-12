from rmdn_hri.dataset import UnimanualDataset

import torch
import numpy as np

import argparse
import os

import rospy
from geometry_msgs.msg import Quaternion, Point
from visualization_msgs.msg import MarkerArray, Marker
from moveit_msgs.msg import DisplayRobotState
from trajectory_msgs.msg import JointTrajectoryPoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rospy.init_node('visualizer_node')

parser = argparse.ArgumentParser(description='Pepper Teleop Tester')
# Results and Paths
parser.add_argument('--src', type=str, default='/home/vignesh/playground/rmdn_hri/data/alap_dataset_singlehand.npz')
args = parser.parse_args()

dataset = UnimanualDataset(args, train=False)

robot_msg = DisplayRobotState()
robot_msg.state.joint_state.header.frame_id = 'base_footprint'
robot_msg.state.joint_state.name = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']

human_msg = MarkerArray()
lines = []
for i in range(3):
	marker = Marker()
	line_strip = Marker()
	line_strip.ns = marker.ns = "human"
	marker.header.frame_id = line_strip.header.frame_id = 'base_link'
	marker.id = i
	line_strip.id = 3+i
	line_strip.lifetime = marker.lifetime = rospy.Duration(0.5)
	line_strip.frame_locked = marker.frame_locked = False
	line_strip.action = marker.action = Marker.ADD

	marker.type = Marker.SPHERE
	line_strip.type = Marker.LINE_STRIP

	line_strip.color.a = marker.color.a = line_strip.color.r = marker.color.g = 1
	line_strip.color.g = line_strip.color.b = marker.color.b = marker.color.r = 0
	marker.scale.x = marker.scale.y = marker.scale.z = 0.075
	line_strip.scale.x = 0.04

	line_strip.pose.orientation = marker.pose.orientation = Quaternion(x=0, y=0, z=0, w=1)

	line_strip.points = [Point(), Point()]

	human_msg.markers.append(marker)
	lines.append(line_strip)
lines = lines[:-1]
human_msg.markers = human_msg.markers + lines

rate = rospy.Rate(30)
robot_pub = rospy.Publisher('display_robot_state', DisplayRobotState, queue_size=10)
human_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)

for i, (xh, xr) in enumerate(dataset):
	seq_len = xh.shape[0]
	for t in range(seq_len):
		for j in range(3):
			human_msg.markers[j].pose.position.x = xh[t, 3*j]
			human_msg.markers[j].pose.position.y = xh[t, 3*j + 1]
			human_msg.markers[j].pose.position.z = xh[t, 3*j + 2]
			if j>0:
				line_idx = 2 + j
				print(j, line_idx)
				human_msg.markers[line_idx].points[0].x = xh[t, 3*j]
				human_msg.markers[line_idx].points[0].y = xh[t, 3*j + 1]
				human_msg.markers[line_idx].points[0].z = xh[t, 3*j + 2]
				human_msg.markers[line_idx].points[1].x = xh[t, 3*(j-1)]
				human_msg.markers[line_idx].points[1].y = xh[t, 3*(j-1) + 1]
				human_msg.markers[line_idx].points[1].z = xh[t, 3*(j-1) + 2]

		robot_msg.state.joint_state.position = xr[t].tolist() + [1.57072, 1.57072, 0.0087, 0., 0.]
		
		stamp = rospy.Time.now()
		for j in range(len(human_msg.markers)):
			human_msg.markers[j].header.stamp = stamp
		robot_msg.state.joint_state.header.stamp = stamp

		robot_pub.publish(robot_msg)
		human_pub.publish(human_msg)
		rate.sleep()
		if rospy.is_shutdown():
			break
	if rospy.is_shutdown():
		break
