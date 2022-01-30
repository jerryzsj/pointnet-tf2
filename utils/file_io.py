#!/usr/bin/env python3

'''
Code developed by Senjing Zheng (senjing.zheng[at]gmail.com)
'''

import os
import sys
import numpy as np
from shutil import copyfile

# np.set_printoptions(precision=3, suppress=True)

###############################
# file i/o functions
# log out string obj
def log_string(fout, out_str):
	print(out_str)
	fout.write(out_str+'\n')
	fout.flush()

# log out list obj
def log_list(fout, out_list):
	for i in out_list:
		fout.write(str(i)+' ')
	fout.write('\n')
	fout.flush()

# log out value obj
def log_value(fout, out_value):
	fout.write(str(out_value)+'\n')
	fout.flush()
###############################
	

def load_npy(file_dir):
	data = np.load(os.path.join(file_dir, 'data.npy'))
	label = np.loadtxt(os.path.join(file_dir, 'label.dat'), dtype=int)
	return data, label

def read_txt(filedir, fmt='txt'):
	data = []
	data_fread = open(filedir)
	line = data_fread.readline()
	while line:
		if fmt=='int':
			data.append([int(l) for l in line.split()])
		if fmt=='float':
			# for l in line.split():
			# for l in line.split():
			# 	if l.isnumeric():
			# 		data.append(float(l))
			data.append([l for l in line.split() if l.isnumeric()])
		if fmt=='txt':
			data.append([l for l in line.split()])
		line = data_fread.readline()
	return np.asarray(data)

def line_to_float(line):
	return(np.asarray([float(l) for l in line.split()]))

if __name__ == '__main__':
	
	exp_name = 'RBF_CleanYCBSTDH20EPOCH20K'

	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	SAVE_BASE_DIR = os.path.join(BASE_DIR, exp_name)
	if not os.path.exists(SAVE_BASE_DIR): os.makedirs(SAVE_BASE_DIR)

	copyfile('file_io.py', os.path.join(SAVE_BASE_DIR, 'file_io.py'))
	copyfile('commonsettings.par', os.path.join(SAVE_BASE_DIR, 'commensettings.par'))
	copyfile('datasettings.par', os.path.join(SAVE_BASE_DIR, 'datasettings.par'))
	copyfile('nnsettings.par', os.path.join(SAVE_BASE_DIR, 'nnsettings.par'))
	copyfile('solutionsettings.par', os.path.join(SAVE_BASE_DIR, 'solutionsettings.par'))
	copyfile(exp_name+'.txt', os.path.join(SAVE_BASE_DIR, exp_name+'.txt'))
	
	parm_f = open(os.path.join(SAVE_BASE_DIR, 'exp_parms.dat'), 'w+')
	parm_f.write('train_acc per 10 epochs\n')
	parm_f.write('eval_acc per 100 epochs\n')
	parm_f.flush()
	parm_f.close()

	with open(exp_name+'.txt') as f:
		alist = [line.rstrip() for line in f]

	# start_idx: 7
	# first exp: idx: 7-1108
	# total num_idx: 1101
	# eval: per 11
	# EXP gap: 1112
	num_exp = 10
	start_idx = 7
	idx_gap = 1102 + 1100*1
	exp_gap = 1111 + 1100*1
	
	eval_acc_list = []

	eval_acc = []

	for exp_idx in range(num_exp):
		log_dir = os.path.join(SAVE_BASE_DIR, str(exp_idx))
		if not os.path.exists(log_dir): os.makedirs(log_dir)

		t_acc_list = []

		e_acc_list = []
		for i in range(start_idx,start_idx+idx_gap):
			if (i-start_idx-1)%(11)==0 or i==start_idx+idx_gap-1:
				e_acc_list.append(line_to_float(alist[i]))
			else:
				t_acc_list.append(line_to_float(alist[i]))

		t_acc_list = (np.asarray(t_acc_list))[:, 2:3]
		e_acc_list = (np.asarray(e_acc_list))[:, :3]

		np.savetxt(os.path.join(log_dir, 'train_acc_log.txt'), t_acc_list, fmt='%.6f')
		np.savetxt(os.path.join(log_dir, 'eval_acc_log.txt'), e_acc_list, fmt='%.6f')
		# np.savetxt(os.path.join(log_dir, 'train_acc_log.txt'), t_acc_list)

		start_idx+=exp_gap

		eval_acc.append(np.mean(e_acc_list, axis=1)[-1])

	np.savetxt(os.path.join(SAVE_BASE_DIR, 'eval_acc.txt'), np.asarray(eval_acc), fmt='%.6f')

	# np.savetxt('./MLP_Clean/Clean/train_acc_list.txt', train_acc)
	# np.savetxt('./MLP_Clean/Clean/eval_acc_list.txt', eval_acc)
	# np.savetxt('./MLP_Clean/Clean/eval_acc.txt', mean_eval_acc)
	# np.savetxt('./MLP_Clean/Clean/train_acc.txt', train_acc[-1, -1])