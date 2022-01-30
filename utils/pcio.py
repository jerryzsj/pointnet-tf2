import os
import sys
import numpy as np
# import h5py
import open3d
from hashlib import md5
import binascii


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')


def load_ycb_pcd(filedir, filelist):
	forder_1 = get_filelist(filedir, 'filelist')

	box_forder = []
	cylinder_forder = []
	sphere_forder = []

	box_forder.extend(get_filelist(forder_1[0], filelist))
	box_object_list = get_objectlist(forder_1[0], filelist)
	
	cylinder_forder.extend(get_filelist(forder_1[1], filelist))
	cylinder_object_list = get_objectlist(forder_1[1], filelist)
	
	sphere_forder.extend(get_filelist(forder_1[2], filelist))
	sphere_object_list = get_objectlist(forder_1[2], filelist)

	box_list=[]
	cylinder_list=[]
	sphere_list=[]

	object_total = []

	for i in range(len(box_forder)):
		box_list.extend(get_filelist(box_forder[i], 'filelist'))
		object_total.extend([box_object_list[i]]*50)
	for i in range(len(cylinder_forder)):
		cylinder_list.extend(get_filelist(cylinder_forder[i], 'filelist'))
		object_total.extend([cylinder_object_list[i]]*50)
	for i in range(len(sphere_forder)):
		sphere_list.extend(get_filelist(sphere_forder[i], 'filelist'))
		object_total.extend([sphere_object_list[i]]*50)

	label = np.full(len(box_list), 0, dtype=int)
	label = np.append(label, (np.full(len(cylinder_list), 1, dtype=int)), axis=0)
	label = np.append(label, (np.full(len(sphere_list), 2, dtype=int)), axis=0)

	pcd_list = []
	pcd_list.extend(box_list)
	pcd_list.extend(cylinder_list)
	pcd_list.extend(sphere_list)

	np_pcd = []
	for i in range(len(pcd_list)):
		np_pcd.append(load_pcd(pcd_list[i]))
	data = np.array(np_pcd)

	return data, label, np.array(object_total)


# def load_partial_ycb_pcd(filedir, filelist):
# 	forder_1 = get_filelist(filedir, "filelist")
# 	box_forder = []
# 	cylinder_forder = []
# 	sphere_forder = []

# 	box_forder.extend(get_filelist(forder_1[0], filelist))
# 	box_object_list = get_objectlist(forder_1[0], filelist)
	
# 	cylinder_forder.extend(get_filelist(forder_1[1], filelist))
# 	cylinder_object_list = get_objectlist(forder_1[1], filelist)
	
# 	sphere_forder.extend(get_filelist(forder_1[2], filelist))
# 	sphere_object_list = get_objectlist(forder_1[2], filelist)
	
# 	box_list=[]
# 	cylinder_list=[]
# 	sphere_list=[]
# 	object_total = []

# 	for i in range(len(box_forder)):
# 		box_list.extend(get_filelist(box_forder[i], 'filelist'))
# 		object_total.extend([box_object_list[i]]*50)
# 	for i in range(len(cylinder_forder)):
# 		cylinder_list.extend(get_filelist(cylinder_forder[i], 'filelist'))
# 		object_total.extend([cylinder_object_list[i]]*50)
# 	for i in range(len(sphere_forder)):
# 		sphere_list.extend(get_filelist(sphere_forder[i], 'filelist'))
# 		object_total.extend([sphere_object_list[i]]*50)
	
# 	label = np.full(len(box_list), 0, dtype=int)
# 	label = np.append(label, (np.full(len(cylinder_list), 1, dtype=int)), axis=0)
# 	label = np.append(label, (np.full(len(sphere_list), 2, dtype=int)), axis=0)

# 	pcd_list = []
# 	pcd_list.extend(box_list)
# 	pcd_list.extend(cylinder_list)
# 	pcd_list.extend(sphere_list)

# 	np_pcd = []
# 	for i in range(len(pcd_list)):
# 		np_pcd.append(load_pcd(pcd_list[i]))
# 	data = np.array(np_pcd)
# 	return data, label, np.array(object_total)



def load_shapes_pcd(filedir, filelist):
	forder_1 = get_filelist(filedir, filelist)
	box_list = []
	cylinder_list = []
	sphere_list = []
	for i in range(len(forder_1)):
		box_list.extend(get_filelist(forder_1[i]+'box/', 'filelist'))
		cylinder_list.extend(get_filelist(forder_1[i]+'cylinder/', 'filelist'))
		sphere_list.extend(get_filelist(forder_1[i]+'sphere/', 'filelist'))
	
	# print(cylinder_list)

	label = np.full(len(box_list), 0, dtype=int)
	label = np.append(label, (np.full(len(cylinder_list), 1, dtype=int)), axis=0)
	label = np.append(label, (np.full(len(sphere_list), 2, dtype=int)), axis=0)

	pcd_list = []
	pcd_list.extend(box_list)
	pcd_list.extend(cylinder_list)
	pcd_list.extend(sphere_list)

	np_pcd = []
	for i in range(len(pcd_list)):
		np_pcd.append(load_pcd(pcd_list[i]))
	data = np.array(np_pcd)

	return data, label


def load_oneforder_pcd(filedir, filelist='filelist', sample_point=1000):
	forder_1 = get_filelist(filedir, filelist)
	forder_2 = []

	forder_2.extend(get_filelist(forder_1[0], 'filelist'))
	label = np.full(len(get_filelist(forder_1[0], 'filelist')), 0, dtype=int)

	for i in range(1,len(forder_1)):
		forder_2.extend(get_filelist(forder_1[i], 'filelist'))
		label = np.append(label, (np.full(len(get_filelist(forder_1[i], 'filelist')), i, dtype=int)), axis=0)

	np_pcd = []
	for i in range(len(forder_2)):
		sample_pcd = load_pcd(forder_2[i])
		# np.random.shuffle(sample_pcd)
		np_pcd.append(sample_pcd)
	data = np.array(np_pcd)

	return data, label


def load_threeforder_pcd(filedir, filelist1='filelist', filelist2='filelist', filelist3='filelist', sample_point=1000):
	label = []
	np_pcd = []
	######################
	# first layer folder:
	folder1 = get_filelist(filedir, filelist1)
	for i, fd1 in enumerate(folder1):
		# print('layer_1:',i)
		# print(fd1)
		######################
		# first layer folder:
		folder2 = get_filelist(fd1, filelist2)
		for j, fd2 in enumerate(folder2):
			# print('layer_2:',j)
			# print(fd2)
			######################
			# first layer folder:
			folder3 = get_filelist(fd2, filelist3)
			for k, fd3 in enumerate(folder3):
				# print('layer_3', k)
				# print(fd3)
				np_pcd.append(load_pcd(fd3))
				label.append(i)
	# np_pcd=np.array(np_pcd)
	# print(label)
	# print(np.array(np_pcd).shape)
	return(np.array(np_pcd), label)




def load_npy(file_dir):
	data = np.load(os.path.join(file_dir, 'data.npy'))
	label = np.loadtxt(os.path.join(file_dir, 'label.dat'), dtype=int)
	# if check_npy_md5(file_dir, data):
	# 	return data, label
	return data, label

def load_partid_dat(file_dir, filelist, sample_times=10, sample_point=1000):
	forder_1 = get_filelist(filedir, 'filelist')
	forder_2 = []

	forder_2.extend(get_filelist(forder_1[0], filelist))
	label = np.full(len(get_filelist(forder_1[0], filelist))*sample_times, 0, dtype=int)

	for i in range(1,len(forder_1)):
		forder_2.extend(get_filelist(forder_1[i], filelist))
		label = np.append(label, (np.full(len(get_filelist(forder_1[i], filelist))*sample_times, i, dtype=int)), axis=0)

	np_pcd = []
	for i in range(len(forder_2)):
		sample_pcd = loadtxt(forder_2[i][:-3]+'.dat')
		np.random.shuffle(sample_pcd)
		idx_begin = 0
		for j in range(sample_times):
			idx_end=idx_begin+sample_point
			np_pcd.append(sample_pcd[idx_begin:idx_end])
			idx_begin=idx_end+1
	data = np.array(np_pcd)

	data = np.load(os.path.join(file_dir, 'data.npy'))
	label = np.loadtxt(os.path.join(file_dir, 'label.dat'), dtype=int)
	if check_npy_md5(file_dir, data):
		return data, label
	return data, label

def load_objectlist(file_dir):
	object_list = np.loadtxt(os.path.join(file_dir, 'object_list.dat'), dtype=str)
	return object_list


def save_npy(data, label, file_dir):
	np.save(os.path.join(file_dir, 'data.npy'), data)
	np.savetxt(os.path.join(file_dir, 'label.dat'), label, fmt='%d')
	# m = md5()
	# m.update(data.data)
	# md5_v = m.hexdigest()
	
	# f=open(os.path.join(file_dir, 'md5.dat'), 'w+')
	# f.write(md5_v)
	# f.close()
	return

def check_npy_md5(file_dir, data):
	m_input = md5()
	m_input.update(data.data)
	md5_input = m_input.hexdigest()
	
	f=open(os.path.join(file_dir, 'md5.dat'), 'r')
	if f.mode == 'r':
		md5_abs = f.read()

	if md5_input==md5_abs:
		print("Good md5 value of input data")
		return True
	print("Wrong md5 value of input data")
	return False


def move_to_origin(np_pcd):
	mx = np.mean(np_pcd, axis=0)
	np_pcd = np_pcd - mx
	return np_pcd

def move_to_origin_batch(np_pcd):
	mv_pcd = []
	for i in np_pcd:
		mv_pcd.append(move_to_origin(i))
	return np.array(mv_pcd)


def move_to_ws(np_pcd, x=0.1, y=0.1, z=0.1):
	bbox_ = bbox_nppcd(np_pcd)
	move_ = (bbox_[0]-x, bbox_[2]-y, bbox_[4]-z)
	np_pcd = np_pcd - move_
	return np_pcd

def move_to_ws_batch(np_pcd, x=0.1, y=0.1, z=0.1):
	mv_pcd = []
	for i in np_pcd:
		mv_pcd.append(move_to_ws(i, x, y, z))
	return np.array(mv_pcd)

def norm_nppcd(np_pcd, scale=1.0):
	box_ = bbox_nppcd(np_pcd)
	max_ = np.sqrt(box_[1]**2 + box_[3]**2 + box_[5]**2) * scale
	return np_pcd/max_


def norm_nppcd_batch(np_pcd, scale=1.0):
	norm_pcd = []
	for pcd in np_pcd:
		norm_pcd.append(norm_nppcd(pcd, scale))
	return np.array(norm_pcd)


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


def bbox_nppcd(np_pcd):
	bbox = np.min(np_pcd[:,0]), np.max(np_pcd[:,0]), np.min(np_pcd[:,1]), np.max(np_pcd[:,1]), np.min(np_pcd[:,2]), np.max(np_pcd[:,2])	
	return bbox

def load_pcd(filename):
	pcd = open3d.io.read_point_cloud(filename)
	np_pcd = np.asarray(pcd.points)
	return np_pcd

def load_open3d_pcd(filename):
	pcd = open3d.io.read_point_cloud(filename)
	return pcd

def save_pcd(np_pcd, filename, filedir):
	if not os.path.exists(filedir):
		os.makedirs(filedir)
	# pcd = open3d.PointCloud()
	# pcd.points = open3d.Vector3dVector(np_pcd)
	# open3d.write_point_cloud('./' +filedir + '/' + filename + '.pcd', pcd)
	save_dir=os.path.join(filedir, filename+'.pcd')
	save_nppcd(np_pcd, save_dir)
	return

def np_to_pcd(np_pcd):
	pcd = open3d.PointCloud()
	pcd.points = open3d.Vector3dVector(np_pcd)
	return pcd

def nppcd_to_open3d(np_pcd):
	pcd = open3d.PointCloud()
	pcd.points = open3d.Vector3dVector(np_pcd)
	return pcd

def save_pcd_dir(np_pcd, filename, filedir):
	if not os.path.exists(filedir):
		os.makedirs(filedir)
	save_dir = os.path.join(filedir, str(filename) + '.pcd')
	save_nppcd(np_pcd, save_dir)
	return

def save_nppcd(np_pcd, save_dir, save_normals=False, save_xyz=False):
	# if not os.path.exists(filedir):
	# 	os.makedirs(filedir)
	# filename = str(filename) + '.pcd'
	# save_dir = os.path.join(filedir, filename)
	f=open(save_dir,'w+')
	if not save_xyz:
		f.write("# .PCD v0.7 - Point Cloud Data file format\n")
		f.write("VERSION 0.7\n")
		if save_normals:
			f.write("FIELDS x y z normal_x normal_y normal_z\n")
			f.write("SIZE 4 4 4 4 4 4\n")
			f.write("TYPE F F F F F F\n")
			f.write("COUNT 1 1 1 1 1 1\n")
		else:
			f.write("FIELDS x y z\n")
			f.write("SIZE 4 4 4\n")
			f.write("TYPE F F F\n")
			f.write("COUNT 1 1 1\n")
		f.write("WIDTH "+str(np_pcd.shape[0])+"\n")
		f.write("HEIGHT 1\n")
		f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
		f.write("POINTS "+str(np_pcd.shape[0])+"\n")
		f.write("DATA ascii\n")
		f.close()
		save_points(np_pcd, save_dir)
	else:
		f.close()
		save_points(np_pcd, save_dir)


def save_points(np_pcd, save_dir):
	f_ = open(save_dir, 'a')
	for point in np_pcd:
		f_.write( str(np.float32(point[0])) + ' ' + str(np.float32(point[1])) + ' ' + str(np.float32(point[2])) + '\n')
	f_.close()



def show_nppcd(np_pcd):
	pcd = open3d.geometry.PointCloud()
	pcd.points = open3d.utility.Vector3dVector(np_pcd)
	open3d.visualization.draw_geometries_with_vertex_selection([pcd])
	# open3d.visualization.VisualizerWithKeyCallback([pcd])
	return


def show_nppcd_list(np_pcd_list):
	pcd_list = []
	for i in range(len(np_pcd_list)):
		pcd = open3d.geometry.PointCloud()
		pcd.points = open3d.utility.Vector3dVector(np_pcd_list[i])
		pcd_list.append(pcd)
	open3d.visualization.draw_geometries(pcd_list)
	# open3d.visualization.VisualizerWithKeyCallback(pcd_list)
	# print((pcd_list))

def get_filelist(filedir, listname='filelist'):
	filelist=[]
	for line in open(os.path.join(filedir ,listname)):
		# if os.path.join(filedir, line.rstrip()) == '/':
		# 	return filelist
		filelist.append(os.path.join(filedir, line.rstrip()))
	return filelist

def get_shapelist(filedir):
	shapelist=[]
	for line in open(filedir + "/shapelist"):
		shapelist.append(line.rstrip())
	return shapelist

def get_filename(filedir, listname):
	filename=[]
	for line in open(os.path.join(filedir ,listname)):
		filename.append(line.rstrip())
	return filename

def get_listshape(filelist):
	listshape = np.array(np.shape(filelist))
	return listshape


def get_objectlist(filedir, listname):
	objectlist=[]
	for line in open(filedir + "/" +listname):
		a = line.rstrip('\n')
		a = a[:-4]
		objectlist.append(a)
	return objectlist


def load_data(file_dir, listname):
	filelist_1 = get_filelist(file_dir, listname)
	filelist_2 = []

	data = []
	label = []

	for idx in range(len(filelist_1)):
		filelist_2.append(get_filelist(filelist_1[idx], 'filelist'))
	
	for data_type in range(len(filelist_1)):
		for object_class in range(len(filelist_2[data_type])):
			filelist_3 = (get_filelist(filelist_2[data_type][object_class], 'filelist'))
			for object_idx in range(len(filelist_3)):
				label.append(object_class)
				data.append(load_pcd(filelist_3[object_class]))
			
	data = np.array(data, np.float64)
	label = np.array(label)
	print("***************************************")
	print("Loaded data shape",data.shape)
	print("Loaded label shape",label.shape)
	print("***************************************")
	return data, label


def load_data_ycb(file_dir, listname):
	filelist_1 = get_filelist(file_dir, listname)
	filelist_2 = []
	filelist_3 = []
	labellist = []
	for idx in range(len(filelist_1)):
		filelist_2.append(get_filelist(filelist_1[idx], 'filelist'))

	for i in range(len(filelist_1)):
		for j in range(len(filelist_2[i])):
			filelist_3.append(get_filelist(filelist_2[i][j], 'filelist'))
			labellist.append(i)

	data = []
	label = []

	for i in range(len(filelist_3)):
		for j in range(len(filelist_3[i])):
			data.append(load_pcd(filelist_3[i][j]))
			if i<12:
				label.append(0)
			elif i<24:
				label.append(1)
			else:
				label.append(2)
			
	data = np.array(data, np.float64)
	label = np.array(label)
	print("***************************************")
	print("Loaded data shape",data.shape)
	print("Loaded label shape",label.shape)
	print("***************************************")
	return data, label


def sample_numbatch(np_pcd, num_batch):
	idx = np.arange(0, np_pcd.shape[0])
	np.random.shuffle(idx)
	rand_pcd = np_pcd[idx]
	pcd_size = np_pcd.shape[0]
	batchsize = pcd_size // num_batch
	out_pcd = []
	for batch_idx in range(num_batch):
		start_idx = batch_idx * batchsize
		end_idx = (batch_idx+1) * batchsize
		out_pcd.append(rand_pcd[start_idx:end_idx, :])
	return np.array(out_pcd)


def sample_numpoint(np_pcd, num_point, num_batch):
	idx = np.arange(0, np_pcd.shape[0])
	np.random.shuffle(idx)
	rand_pcd = np_pcd[idx]
	pcd_size = np_pcd.shape[0]
	batchsize = num_point
	out_pcd = []

	if (pcd_size // num_point)<num_batch:

		sub_batch = pcd_size // num_point
		num_sample = num_batch // (pcd_size // num_point)

		for sample_idx in range(num_sample+1):
			np.random.shuffle(idx)
			rand_pcd = np_pcd[idx]
			for batch_idx in range(sub_batch):
				start_idx = batch_idx * batchsize
				end_idx = (batch_idx+1) * batchsize
				out_pcd.append(rand_pcd[start_idx:end_idx, :])

	else:
		for batch_idx in range(num_batch):
			start_idx = batch_idx * batchsize
			end_idx = (batch_idx+1) * batchsize
			out_pcd.append(rand_pcd[start_idx:end_idx, :])

	return np.array(out_pcd)


## Sampling ycb point cloud files (.ply) in folder ../data/ycb/ by fixed number of batches
## Data folder containing just two layers
def exe_sample_fix_numbatch_2layers(num_batch=100, data_forder='ycb/'):	
	data_dir = DATA_DIR
	data_dir = os.path.join(data_dir, data_forder)
	filelist = get_filelist(data_dir)
	shapelist = get_shapelist(data_dir)
	filedir = []
	filename = []
	for idx in range(len(filelist)):
		filedir.append(get_filelist(filelist[idx]))
		filename.append(get_filename(filelist[idx]))
	
	filelist_shape = get_listshape(filedir)

	for i in range(filelist_shape[0]):
		for j in range(filelist_shape[1]):
			pcd = load_pcd(filedir[i][j] + '.ply')
			sample = sample_numbatch(pcd, num_batch)
			for k in range(num_batch//2):
				save_pcd(sample[k], str(k), shapelist[i]+ '/' +filename[i][j])
			for k in range(num_batch//2):
				save_pcd(sample[k+50], str(k), 'test/'+shapelist[i]+ '/' +filename[i][j])


## Sampling ycb point cloud files (.ply) in folder ../data/ycb/ by fixed number of points
## Data folder containing just two layers
def exe_sample_fix_numpoint_2layers(num_point = 1000, num_batch=2, data_forder='ycb/'):	
	data_dir = DATA_DIR
	data_dir = os.path.join(data_dir, data_forder)
	filelist = get_filelist(data_dir)
	shapelist = get_shapelist(data_dir)
	filedir = []
	filename = []
	for idx in range(len(filelist)):
		filedir.append(get_filelist(filelist[idx]))
		filename.append(get_filename(filelist[idx]))
	
	filelist_shape = get_listshape(filedir)

	for i in range(filelist_shape[0]):
		for j in range(filelist_shape[1]):
			pcd = load_pcd(filedir[i][j] + '.ply')
			sample = sample_numpoint(pcd, num_point, num_batch)
			for k in range(num_batch//2):
				save_pcd(move_to_origin(sample[k]), str(k), shapelist[i]+ '/' +filename[i][j])
			for k in range(num_batch//2):
				save_pcd(move_to_origin(sample[k+num_batch//2]), str(k), 'test/'+shapelist[i]+ '/' +filename[i][j])



if __name__ == "__main__":
	DATA_DIR = os.path.join(DATA_DIR, 'shapes')
	DATA_DIR = os.path.join(DATA_DIR, 'shapes_meter')
	# filelist = 'filelist'
	# data, label = load_shapes_pcd(DATA_DIR, filelist)

	data2, label2 = load_npy(DATA_DIR)
	# if data.all()==data2.all():
	# 	print("good")
	# m = md5()
	# m.update(data2.all())
	# bm = m.hexdigest()
	# # bm = binascii.hexlify(m)
	# print(bm)

	# # data3, label3 = load_npy(DATA_DIR)
	# # m2 = md5()
	# # m2.update(data2.all())
	# # bm2 = m2.hexdigest()
	# # print(bm2)

	# # f=open(os.path.join(DATA_DIR, 'md5.dat'), 'w+')
	# # f.write(bm2)
	# # f.close()

	# f=open(os.path.join(DATA_DIR, 'md5.dat'), 'r')
	# if f.mode == 'r':
	# 	bm3 = f.read()


	# if bm == bm3:
	# 	print("good")








	# exe_sample_fix_numpoint_2layers()
	# data_forder='ycb/'
	# data_dir = DATA_DIR
	# data_dir = os.path.join(data_dir, data_forder)
	# filelist = get_filelist(data_dir)
	# shapelist = get_shapelist(data_dir)
	# filedir = []
	# filename = []
	# for idx in range(len(filelist)):
	# 	filedir.append(get_filelist(filelist[idx]))
	# 	filename.append(get_filename(filelist[idx]))
	
	# filelist_shape = get_listshape(filedir)

	# for i in range(filelist_shape[0]):
	# 	for j in range(filelist_shape[1]):
	# 		pcd = load_pcd(filedir[i][j] + '.ply')
	# 		print(pcd.shape)
	# pcd = load_pcd('./box/026_sponge/1.pcd')
	# show_nppcd(pcd)
	


	# num_point = 1000
	# num_batch = 2
	# data_forder='ycb/'
	# data_dir = DATA_DIR
	# data_dir = os.path.join(data_dir, data_forder)
	# filelist = get_filelist(data_dir)
	# shapelist = get_shapelist(data_dir)
	# filedir = []
	# filename = []
	# for idx in range(len(filelist)):
	# 	filedir.append(get_filelist(filelist[idx]))
	# 	filename.append(get_filename(filelist[idx]))
	
	# filelist_shape = get_listshape(filedir)

	# for i in range(filelist_shape[0]):
	# 	for j in range(filelist_shape[1]):
	# 		pcd = load_pcd(filedir[i][j] + '.ply') * 10
	# 		sample = sample_numpoint(pcd, num_point, num_batch)
	# 		show_nppcd(sample[0])
	# 		origin_sample = move_to_origin(sample[0])
	# 		show_nppcd(origin_sample)