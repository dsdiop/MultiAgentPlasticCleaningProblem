import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy import interpolate

from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_trajectory_v0(env_map, trajectories):

	plt.style.use('grayscale')

	plt.ion()

	styles = ['solid', 'dashed', 'dashdot', 'dotted']

	env_map = ndimage.binary_dilation(env_map, [[False, True, False], [True, True, True], [False, True, False]])


	n_trajs = int(trajectories.shape[1]/2)

	fig, ax = plt.subplots(1, 1)

	ax.imshow(env_map, cmap='gray')
	ax.get_yaxis().set_visible(False)
	ax.get_xaxis().set_visible(False)

	for idx, a in enumerate(range(0,n_trajs*2,2)):
		xp = trajectories[:, a + 1]
		yp = trajectories[:, a]

		okay = np.where(np.abs(np.diff(xp)) + np.abs(np.diff(yp)) > 0)

		xp = xp[okay]
		yp = yp[okay]

		tck, u = interpolate.splprep([xp, yp], s=0.0)
		x_i, y_i = interpolate.splev(np.linspace(0, 1, 300), tck)

		ax.plot(x_i, y_i, ls=styles[idx], lw=1.5, label = f'Agent {idx+1}')


	plt.legend()
	plt.show()


def plot_path(path: np.ndarray, axs = None, title: str = ''):
	""" Plot the trajectories and the peaks of the trajectories. """

	if axs is None:
		fig, axs = plt.subplots(1,1, figsize=(10,10))

	# Plot the paths of the agents gradually changing the color of the line for every point #
	for i in range(path.shape[0]):
		axs.plot(path[:,1], path[:,0], color=(i/path.shape[0], 0, 1-i/path.shape[0]))
	
	return axs


def plot_trajectory_v1(ax, x, y, z=None, colormap = 'jet', num_of_points = None, linewidth = 1, k = 3, plot_waypoints=False, markersize = 0.5, alpha=1, zorder=1, s=0.0):

	# Remove consecuitve duplicates values of XYZ #
	if z is None:
		path = np.array([x,y]).T
	else:
		path = np.array([x,y,z]).T

	for i in range(path.shape[0]-1, 0, -1):
		if np.array_equal(path[i], path[i-1]):
			path = np.delete(path, i, axis=0)
	
	x = path[:,0]
	y = path[:,1]
	if z is not None:
		z = path[:,2]




	if z is None:
		tck, u = interpolate.splprep([x, y], s=s, k=k)
		x_i, y_i= interpolate.splev(np.linspace(0,1,num_of_points),tck)
		points = np.array([x_i,y_i]).T.reshape(-1,1,2)
		segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
		lc = LineCollection(segments, norm = plt.Normalize(0, 1),cmap=plt.get_cmap(colormap), linewidth=linewidth, alpha=alpha, zorder=zorder)
		lc.set_array(np.linspace(0,1,len(x_i)))
		ax.add_collection(lc)
		if plot_waypoints:
			ax.plot(x,y,'.', color = 'black', markersize = markersize, zorder=zorder+1)
	else:
		tck, u =interpolate.splprep([x, y, z], s=0.0)
		x_i, y_i, z_i= interpolate.splev(np.linspace(0,1,num_of_points), tck)
		points = np.array([x_i, y_i, z_i]).T.reshape(-1,1,3)
		segments = np.concatenate([points[:-2], points[1:-1], points[2:]], axis=1)
		lc = Line3DCollection(segments, norm = plt.Normalize(0, 1),cmap=plt.get_cmap(colormap), linewidth=linewidth)
		lc.set_array(np.linspace(0,1,len(x_i)))
		ax.add_collection(lc)
		ax.scatter(x,y,z,'k')
		if plot_waypoints:
			ax.plot(x,y,'kx')


