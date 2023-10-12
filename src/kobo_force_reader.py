#!/usr/bin/python

import rospy
import numpy as np
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import Float32

rospy.init_node('reset_kobo')
force_pub = rospy.Publisher('/force_norm', Float32, queue_size=10)

def callback(msg:WrenchStamped):
    force_pub.publish(Float32(np.linalg.norm(np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]))))

force_sub = rospy.Subscriber("/panda_dual/panda_right_state_controller/F_ext", WrenchStamped, callback)
rospy.spin()
