import matplotlib.pyplot as plt
import pbdlib as pbd

from visualization_msgs.msg import MarkerArray, Marker
import rospy

from utils.helper import *
from utils.nuitrack import *

point_color = (0, 255, 0)
line_color = (0, 0, 225)


def visualize_skeleton(data, variant=None, action=None):
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	# plt.ion()

	print('data',data.shape)
	# center_mean = (data[0, :, :3].sum(axis=0) + data[0, :, 3:].sum(axis=0))/(2*data.shape[1])
	# print(center_mean)
	# data[:, :, :3] = data[:, :, :3] - center_mean
	# data[:, :, 3:] = data[:, :, 3:] - center_mean
	ax.view_init(0, -0)
	# ax.grid(False)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	# ax.set_axis_bgcolor('white')w
	for frame_idx in range(data.shape[0]):
		ax.cla()
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		ax.set_facecolor('none')
		ax.set_xlim3d([-0.9, 0.1])
		ax.set_ylim3d([-0.1, 0.9])
		ax.set_zlim3d([-0.65, 0.35])
		ax.set_title("Frame: {}".format(frame_idx))

		# ax.axis('off')
		if variant is not None and action is not None:
			# ax.set_title('_'.join(variant.split('/')[0]) + " " + action)
			ax.set_title(variant + " " + action)

		x = data[frame_idx, :, 0]
		y = data[frame_idx, :, 1]
		z = data[frame_idx, :, 2]
		ax.scatter(x, y, z, color='r', marker='o')

		x = data[frame_idx, :, 3]
		y = data[frame_idx, :, 4]
		z = data[frame_idx, :, 5]
		ax.scatter(x, y, z, color='b', marker='o')
		plt.pause(0.01)
		if not plt.fignum_exists(1):
			break
	
	plt.ioff()
	plt.show()

def prepare_axes(ax):
	ax.cla()
	# ax.view_init(15, 160)
	ax.set_xlim3d([-0.9, 0.1])
	ax.set_ylim3d([-0.1, 0.9])
	ax.set_zlim3d([-0.65, 0.35])
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

def plot_skeleton(ax, skeleton):
	for i in range(len(connections)):
		bone = connections[i]
		ax.plot(skeleton[[joints_idx[bone[0]]-1, joints_idx[bone[1]]-1], 0], skeleton[[joints_idx[bone[0]]-1, joints_idx[bone[1]]-1], 1], skeleton[[joints_idx[bone[0]]-1, joints_idx[bone[1]]-1], 2], 'r-', linewidth=5)
	ax.scatter(skeleton[:-1, 0], skeleton[:-1, 1], skeleton[:-1, 2], c='g', marker='o', s=100)
	ax.scatter(skeleton[-1:, 0], skeleton[-1:, 1], skeleton[-1:, 2], c='g', marker='o', s=200)

def plot_pbd(ax, model, alpha_hsmm=None, dims = slice(0,3), color='red'):
	if alpha_hsmm is None:
		pbd.plot_gmm3d(ax, model.mu[:,dims], model.sigma[:,dims,dims], color=color, alpha=0.3)
	else:
		for i in range(model.nb_states):
			pbd.plot_gauss3d(ax, model.mu[i,dims], model.sigma[i,dims,dims],
						n_points=20, n_rings=15, color='blue', alpha=alpha_hsmm[i])

def rviz_gmm3d(model, nstd=3, dims = slice(0,3), rgb = [0,0,1], frame_id='base_footprint'):
	markerarray_msg = MarkerArray()
	T = np.eye(4)
	for i in range(model.nb_states):
		marker = Marker()
		marker.id = i
		marker.lifetime = rospy.Duration(20)
		marker.frame_locked = True
		marker.action = Marker.ADD
		marker.type = Marker.SPHERE
		marker.color.a = 0.5
		marker.color.r = rgb[0]
		marker.color.g = rgb[1]
		marker.color.b = rgb[2]
		marker.header.frame_id = frame_id

		eigvals, eigvecs = np.linalg.eig(model.sigma[i,dims,dims])
		eigvecs[:, 0] /= np.linalg.norm(eigvecs[:, 0])
		eigvecs[:, 1] /= np.linalg.norm(eigvecs[:, 1])
		eigvecs[:, 2] /= np.linalg.norm(eigvecs[:, 2])
		# Ensure right handed notation of rotation matrix
		if np.cross(eigvecs[:, 0], eigvecs[:, 1]).dot(eigvecs[:, 2]) < 0:
			eigvecs[:, [0,1]] = eigvecs[:, [1,0]]
			eigvals[[0,1]] = eigvals[[1,0]]

		marker.scale.x, marker.scale.y, marker.scale.z = nstd * np.sqrt(eigvals)
		T[:3,:3] = eigvecs
		T[:3,3] = model.mu[i, dims]
		marker.pose = mat2Pose(T)
		
		markerarray_msg.markers.append(marker)
	return markerarray_msg
