import numpy as np

joints = ["none", "head", "neck", "torso", "waist", "left_collar", "left_shoulder", "left_elbow", "left_wrist", "left_hand", "right_collar", "right_shoulder", "right_elbow", "right_wrist", "right_hand", "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"]
joints_idx = {joints[i]:i for i in range(len(joints))}
connections = [
	["head", "neck"],
	["neck", "torso"],
	["torso", "waist"],
	["neck", "left_collar"],
	["left_collar", "left_shoulder"],
	["left_shoulder", "left_elbow"],
	["left_elbow", "left_wrist"],
	["left_wrist", "left_hand"],
	["neck", "right_collar"],
	["right_collar", "right_shoulder"],
	["right_shoulder", "right_elbow"], 
	["right_elbow", "right_wrist"],
	["right_wrist", "right_hand"]
]

intrinsics_horizontal = {
	(1920,1080): np.array([
					[963.013, 0., 955.925],
					[0., 963.013, 535.814],
					[0., 0., 1.]
				]), 
	(1280,720) : np.array([
					[642.009, 0., 637.284],
					[0., 642.009, 357.209],
					[0., 0., 1.]
				]),
	(848,480): np.array([
					[616.640625, 0.0, 428.6637268066406],
					[0.0, 616.4426879882812, 243.40573120117188],
					[0., 0., 1.]
				]),
	(640,480): np.array([
					[321.004, 0., 318.642],
					[0., 321.004, 178.605],
					[0., 0., 1.]
				])
}

# # since the camera is vertically placed, cx and cy are interchanged
intrinsics_vertical = {
	(1920,1080): np.array([
					[963.013, 0., 535.814],
					[0., 963.013, 955.925],
					[0., 0., 1.]
				]), 
	(1280,720) : np.array([
					[642.009, 0., 357.209],
					[0., 642.009, 637.284],
					[0., 0., 1.]
				]),
	(848,480): np.array([
					[616.4426879882812, 0.0, 243.40573120117188],
					[0.0, 616.640625, 428.6637268066406],
					[0., 0., 1.]
				]),
	(640,480): np.array([
					[321.004, 0., 178.605],
					[0., 321.004, 318.642],
					[0., 0., 1.]
				])
}