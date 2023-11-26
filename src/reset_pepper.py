#!/usr/bin/python

import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pepper_controller_server.srv import JointTarget
import sys

rospy.init_node('pepper_reset_node')
rospy.wait_for_service('/pepper_dcm/RightArm_controller/goal')
send_target = rospy.ServiceProxy('/pepper_dcm/RightArm_controller/goal', JointTarget)
rate = rospy.Rate(100)
joint_trajectory = JointTrajectory()
joint_trajectory.header.frame_id = "base_footprint"
joint_trajectory.joint_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", 'RHand']
joint_trajectory.points.append(JointTrajectoryPoint())
joint_trajectory.points[0].effort = np.zeros(len(joint_trajectory.joint_names)).tolist()
joint_trajectory.points[0].effort[0] = 0.3
joint_trajectory.points[0].positions = [1.5708, -0.109, 0.7854, 0.009, 1., float(sys.argv[-1])] # default standing angle values with the hand openness DoF as input
joint_trajectory.header.stamp = rospy.Time.now()
send_target(joint_trajectory)
rospy.Rate(10).sleep()
joint_trajectory.header.stamp = rospy.Time.now()
send_target(joint_trajectory)
rospy.Rate(10).sleep()
joint_trajectory.header.stamp = rospy.Time.now()
send_target(joint_trajectory)
rospy.Rate(10).sleep()
rospy.signal_shutdown('done')