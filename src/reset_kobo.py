#!/usr/bin/python

import rospy
from geometry_msgs.msg import PoseStamped

rospy.init_node('reset_kobo')
goal_pub = rospy.Publisher('/panda_dual/dual_arm_cartesian_impedance_controller/centering_frame_target_pose', PoseStamped, queue_size=10)
goal_msg = PoseStamped()
goal_msg.header.frame_id = 'panda_left_link0'
goal_msg.pose.orientation.x = 0.4618
goal_msg.pose.orientation.y = -0.7329
goal_msg.pose.orientation.z = 0.1907
goal_msg.pose.orientation.w = 0.4618
goal_msg.pose.position.x = 0.336
goal_msg.pose.position.y = -0.349
goal_msg.pose.position.z = 0.121
goal_msg.header.stamp = rospy.Time.now()
goal_pub.publish(goal_msg)
rospy.Rate(1).sleep()
goal_msg.header.stamp = rospy.Time.now()
goal_pub.publish(goal_msg)
rospy.Rate(1).sleep()
rospy.signal_shutdown('Done')
