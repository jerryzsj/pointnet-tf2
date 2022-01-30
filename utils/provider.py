import os
import sys
import numpy as np
from numpy import random as nr
nr.seed()
# import h5py
# import open3d 

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = PROJECT_ROOT

sys.path.append(BASE_DIR)

# Download dataset for point cloud classification
DATA_DIR = os.path.join(BASE_DIR, 'data')
if not os.path.exists(DATA_DIR):
	os.mkdir(DATA_DIR)
# if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
#     www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
#     zipfile = os.path.basename(www)
#     os.system('wget %s; unzip %s' % (www, zipfile))
#     os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
#     os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
	""" Shuffle data and labels.
		Input:
		  data: B,N,... numpy array
		  label: B,... numpy array
		Return:
		  shuffled data, label and shuffle indices
	"""
	idx = np.arange(len(labels))
	np.random.shuffle(idx)
	return data[idx, ...], labels[idx], idx



## functions for angle
def degree_to_radian(degree):
	radian = degree/90*np.pi/2
	return radian

def radian_to_degree(radian):
	degree = radian * 180.0 /math.pi
	return np.float16(degree)

def d_sin(degree):
	if (degree%180==0):
		return 0
	else:
		return(np.sin(degree_to_radian(degree)))

def d_cos(degree):
	if ((degree+90)%180==0):
		return 0
	else:
		return(np.cos(degree_to_radian(degree)))

## Calculate rotation matrix
# input degree for rotating in axis-z, then in axis-x, and finally axis-y
def cal_rotation_matrix_zxy(o_z=0, o_x=0, o_y=0):
	
	s_z = d_sin(o_z)
	c_z = d_cos(o_z)
	r_z = np.array([[c_z, -s_z, 0], [s_z, c_z, 0], [0,0,1]])

	s_x = d_sin(o_x)
	c_x = d_cos(o_x)
	r_x = np.array([[1, 0, 0], [0, c_x, -s_x], [0, s_x, c_x]])

	s_y = d_sin(o_y)
	c_y = d_cos(o_y)
	r_y = np.array([[c_y, 0, s_y], [0, 1, 0], [-s_y, 0, c_y]])

	r_matrix = r_z.dot(r_x).dot(r_y)
	return r_matrix

def get_rotation_matrix_zxy(o_z=0, o_x=0, o_y=0):
	r_matrix = cal_rotation_matrix_zxy(o_z, o_x, o_y)
	return r_matrix

def rotate_point_cloud_full(batch_data):
	# randomly rotate the point clouds with universal aspect
	rotated_data = np.copy(batch_data)
	for i in range(rotated_data.shape[0]):
		rotated_data[i] = np.dot(rotated_data[i],get_rotation_matrix_zxy(nr.random()*360, nr.random()*360, nr.random()*360).T)
	return rotated_data

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
	""" Rotate the point cloud along up direction with certain angle.
		Input:
		  BxNx3 array, original batch of point clouds
		Return:
		  BxNx3 array, rotated batch of point clouds
	"""
	rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
	for k in range(batch_data.shape[0]):
		#rotation_angle = np.random.uniform() * 2 * np.pi
		cosval = np.cos(rotation_angle)
		sinval = np.sin(rotation_angle)
		rotation_matrix = np.array([[cosval, 0, sinval],
									[0, 1, 0],
									[-sinval, 0, cosval]])
		shape_pc = batch_data[k, ...]
		rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
	return rotated_data


def jitter_normed_point_cloud(batch_data, sigma=0.05):
	""" Randomly jitter points. jittering is per point.
		Input:
		  BxNx3 array, original batch of point clouds
		Return:
		  BxNx3 array, jittered batch of point clouds
	"""
	# generate uniform distribution random number between -sigma ~ +sigmas
	B, N, C = batch_data.shape
	jittered_data = (2*sigma) * np.random.rand(B,N,C) - sigma
	jittered_data += batch_data
	return jittered_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
	""" Randomly jitter points. jittering is per point.
		Input:
		  BxNx3 array, original batch of point clouds
		Return:
		  BxNx3 array, jittered batch of point clouds
	"""
	B, N, C = batch_data.shape
	assert(clip > 0)
	jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
	jittered_data += batch_data
	return jittered_data


def getDataFiles(list_filename):
	return [line.rstrip() for line in open(list_filename)]

def getDataDir(list_filename):
	file_dir = [line.rstrip() for line in open(list_filename)]
	folder_dir = os.path.dirname(list_filename)
	data_dir = []
	for i in range(len(file_dir)):
		data_dir.append(os.path.join(folder_dir, file_dir[i]))
	return data_dir

