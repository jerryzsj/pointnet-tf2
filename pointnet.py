""" Implement based on PointNet: https://github.com/charlesq34/pointnet
Author: Charles R. Qi
Date: November 2016

Modified by: Senjing Zheng
Date: January 2022
"""

import argparse
import sys
import os
import importlib
import time
from datetime import date

time_start = time.perf_counter()

import resource
from shutil import copyfile

import math
import numpy as np
from numpy import random as nr
import tensorflow as tf
GPU_DEVICE = tf.test.gpu_device_name()
print('Using GPU: ' + GPU_DEVICE)

PROJECT_DIR = os.getcwd()
BASE_DIR = os.path.dirname(PROJECT_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')

sys.path.append(PROJECT_DIR)
sys.path.append(os.path.join(PROJECT_DIR, 'models'))
sys.path.append(os.path.join(PROJECT_DIR, 'utils'))
import tf_util
import provider
from file_io import *
from email_utils import send_an_email

LOG_BASE_DIR = os.path.join(PROJECT_DIR, 'log')
if not os.path.exists(LOG_BASE_DIR): os.makedirs(LOG_BASE_DIR)

DUMP_BASE_DIR = os.path.join(PROJECT_DIR, 'dump')
if not os.path.exists(DUMP_BASE_DIR): os.makedirs(DUMP_BASE_DIR)

#Global variables/parameters
BASE_LEARNING_RATE = 0.001
GPU_INDEX = 0
MOMENTUM = 0.9  # momemtum for momentum optimizer
FILE_LIST = 'filelist'
BN_INIT_DECAY = 0.5


# Input parameters
parser = argparse.ArgumentParser()
# name of experiment
parser.add_argument('--decay_step', type=int, default=200000, help='[default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='[default: 0.7]')
parser.add_argument('--exp_name', default='0128-code-test-shapes-clean-clean-Epoch10', help='Log dir [default: log]')
# settings for basic hyperparameters
parser.add_argument('--optimizer', default='adam', help='[adam/momentum]')
parser.add_argument('--max_epoch', type=int, default=10, help='Epoch to run [default: 250]')
parser.add_argument('--num_class', type=int, default=3, help='Number of class')
parser.add_argument('--batch_size', type=int, default=100, help='Batch Size during training [default: 100]')
parser.add_argument('--num_point', type=int, default=1000)
parser.add_argument('--reiteration', type=int, default=10, help='experimental reiteration for statistical purpose')
parser.add_argument('--exp_start_idx', type=int, default=0, help='experimental reiteration for statistical purpose')
# settings for datasets
parser.add_argument('--train_type', default='shapes', help='Dataset type [shapes/ycb/mechnet/normalized]')
parser.add_argument('--train_name', default='shapes_luca_clean_norm', help='Data forder [shapes_luca_error_norm/shapes_all]') 
parser.add_argument('--test_type', default='shapes')
parser.add_argument('--test_name', default='shapes_luca_clean_norm')
# settings for data augmentation
parser.add_argument('--rotate_pcl', default='True', help='rotate pcl before feeding into pointnet')
parser.add_argument('--rotate_type', default='full', help='[full, up]')
parser.add_argument('--jitter_pcl', default='False', help='jitter pcl before feeding into pointnet')
parser.add_argument('--jitter_sigma', type=float, default=0.01, help='+-sigma for each x,y,z')

# settings for EA scheme
parser.add_argument('--ea_scheme', default='False')
parser.add_argument('--ea_type', default='cannt', help='EA scheme: [None, origin, batch, history_selection, cannt]')

# parms for GenaticAlgorithm
parser.add_argument('--selection_scheme', default='fitness_proportionate_selection', help='selection_scheme: [rank_selection, fitness_proportionate_selection]')
parser.add_argument('--replace_scheme', default='replace_with_children', help='replace_with_rank: [replace_with_children, replace_with_rank]')

# parms for Population
parser.add_argument('--num_people', type=int, default=2)
parser.add_argument('--num_child', type=int, default=1)

# parms for GenePool
parser.add_argument('--total_num_genes', type=int, default=591)
parser.add_argument('--gene_pool_initiated_fitness', type=float, default=1.0)
parser.add_argument('--gene_pool_memory_len', type=int, default=10)
parser.add_argument('--history_weighted_scheme', default='True')
parser.add_argument('--long_weight', type=float, default=0.8)
parser.add_argument('--short_weight', type=float, default=0.2)
parser.add_argument('--forget_scheme', default='True')
parser.add_argument('--forget_threshold', type=float, default=1.5)

# parms for Individuals
parser.add_argument('--individual_init_scheme', default='full', help='individual_init_scheme: [rendom, full]')
parser.add_argument('--murate_individual', type=float, default=0.3)
parser.add_argument('--murate_genes', type=float, default=0.3)
FLAGS = parser.parse_args()

EXP_START_IDX=FLAGS.exp_start_idx

DECAY_STEP=FLAGS.decay_step
DECAY_RATE=FLAGS.decay_rate
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

OPTIMIZER = FLAGS.optimizer
MAX_EPOCH = FLAGS.max_epoch
NUM_CLASS = FLAGS.num_class
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
# OBJ_ACC_VIS_EPOCH = FLAGS.obj_acc_vis_epoch

MODEL = importlib.import_module('pointnet_cls_'+str(NUM_CLASS))
MODEL_EVAL = importlib.import_module('pointnet_cls_'+str(NUM_CLASS)+'_fpoints')

REITERATION = FLAGS.reiteration
EXP_NAME = FLAGS.exp_name

if FLAGS.rotate_pcl == 'True':
    ROTATE_PCL = True
else:
    ROTATE_PCL = False
ROTATE_TYPE = FLAGS.rotate_type

if FLAGS.jitter_pcl == 'True':
    JITTER_PCL = True
else:
    JITTER_PCL = False

JITTER_SIGMA = FLAGS.jitter_sigma

if FLAGS.ea_scheme=='True':
    EA_SCHEME = True
else:
    EA_SCHEME = False

EA_TYPE = FLAGS.ea_type

SELECTION_SCHEME = FLAGS.selection_scheme
REPLACE_SCHEME = FLAGS.replace_scheme

NUM_PEOPLE = FLAGS.num_people
NUM_CHILD = FLAGS.num_child

TOTAL_NUM_GENES = FLAGS.total_num_genes
GENE_POOL_INITIATED_FITNESS = FLAGS.gene_pool_initiated_fitness
GENE_POOL_MEMORY_LEN = FLAGS.gene_pool_memory_len

if FLAGS.history_weighted_scheme == 'True':
    HISTORY_WEIGHTED_SCHEME=True
else:
    HISTORY_WEIGHTED_SCHEME=False

LONG_WEIGHT = FLAGS.long_weight
SHORT_WEIGHT = FLAGS.short_weight

if FLAGS.forget_scheme == 'True':
    FORGET_SCHEME=True
else:
    FORGET_SCHEME=False

FORGET_THRESHOLD = FLAGS.forget_threshold

INDIVIDUAL_INIT_SCHEME = FLAGS.individual_init_scheme
MURATE_INDIVIDUAL = FLAGS.murate_individual
MURATE_GENES = FLAGS.murate_genes


if EA_SCHEME:
    print('Using EA scheme:', EA_TYPE)
    ea_model_name = 'ea_wrapper_' + EA_TYPE
    EA_MODEL = importlib.import_module(ea_model_name)

else:
    print('Not using EA scheme')

# Load data dir
TRAIN_TYPE = FLAGS.train_type
TRAIN_NAME = FLAGS.train_name
TEST_TYPE = FLAGS.test_type
TEST_NAME = FLAGS.test_name

TRAIN_DATA_DIR = os.path.join(DATA_DIR, TRAIN_TYPE)
TRAIN_DATA_DIR = os.path.join(TRAIN_DATA_DIR, TRAIN_NAME)
TRAIN_DATA_DIR = os.path.join(TRAIN_DATA_DIR, 'train')
TRAIN_DATA, TRAIN_LABEL = load_npy(TRAIN_DATA_DIR)

TEST_DATA=[]
TEST_LABEL=[]

TEST_DATA_DIR = []
TEST_DATA_LIST=[]
TEST_LABEL_LIST=[]
TEST_NAME_LIST = ["3cam_origin_1000_norm_5_error","3cam_origin_1000_norm_10_error","3cam_origin_1000_norm"]

if TEST_TYPE == 'mech12':
    for test_name in TEST_NAME_LIST:
        test_data_dir = os.path.join(DATA_DIR, TEST_TYPE)
        test_data_dir = os.path.join(test_data_dir, test_name)
        test_data_dir = os.path.join(test_data_dir, 'test')
        t_data, t_label = load_npy(test_data_dir)
        TEST_DATA_LIST.append(t_data)
        TEST_LABEL_LIST.append(t_label)

if TEST_TYPE == 'mech12':
    TEST_DATA_DIR = os.path.join(DATA_DIR, TEST_TYPE)
    TEST_DATA_DIR = os.path.join(TEST_DATA_DIR, TEST_NAME_LIST[0])
    TEST_DATA_DIR = os.path.join(TEST_DATA_DIR, 'test')
else:
    TEST_DATA_DIR = os.path.join(DATA_DIR, TEST_TYPE)
    TEST_DATA_DIR = os.path.join(TEST_DATA_DIR, TEST_NAME)
    TEST_DATA_DIR = os.path.join(TEST_DATA_DIR, 'test')
TEST_DATA, TEST_LABEL = load_npy(TEST_DATA_DIR)


print('Train data shape:',TRAIN_DATA.shape)
print('Test data shape:',TEST_DATA.shape)
# END of input parms #
######################

LOG_BASE_DIR = os.path.join(LOG_BASE_DIR, EXP_NAME)
LOG_BASE_DIR = os.path.join(LOG_BASE_DIR, 'log_train')
if not os.path.exists(LOG_BASE_DIR): os.makedirs(LOG_BASE_DIR)
copyfile(os.path.abspath(__file__), os.path.join(LOG_BASE_DIR, 'train.py'))

DUMP_BASE_DIR = os.path.join(DUMP_BASE_DIR, EXP_NAME)
if not os.path.exists(DUMP_BASE_DIR): os.makedirs(DUMP_BASE_DIR)

# tf function: define learning rate decay
def get_learning_rate(batch):
    learning_rate = tf.compat.v1.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

# tf function: define batch-normalisation: decay rate, step
def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(
                        BN_INIT_DECAY,
                        batch*BATCH_SIZE,
                        BN_DECAY_DECAY_STEP,
                        BN_DECAY_DECAY_RATE,
                        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay



def train():
    with tf.Graph().as_default():
        # with GPU acceleration
        with tf.device('/gpu:'+str(GPU_INDEX)):
        # without GPU
        # with tf.device('/cpu:0'):
        # with tf.device('/tpu:0'):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
            
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.compat.v1.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.compat.v1.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.cast(labels_pl, tf.int64))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.compat.v1.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.compat.v1.train.Saver()
            
        # Create a compat.v1.Session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.compat.v1.Session(config=config)

        # Add summary writers
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
                 'labels_pl': labels_pl,
                 'is_training_pl': is_training_pl,
                 'pred': pred,
                 'loss': loss,
                 'train_op': train_op,
                 'merged': merged,
                 'step': batch}

        print('total data:', TOTAL_GENOME)
            
        # run the training
        for epoch in range(MAX_EPOCH):
            log_string(LOG_FOUT,'**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            if not EA_SCHEME:
                count_miss, count_try = train_one_epoch(sess, ops, train_writer)
                validate_one_epoch(sess, ops, test_writer, epoch)
                # _ = update_pred_map(sess, ops, test_writer, epoch)

            if EA_SCHEME and (EA_TYPE=='cannt'):
                # print('running one epoch')
                # get training data-idx
                data_idx = myGA.get_data_idx()
                # express_chromo_distribution, express_chromo_num = myGA.get_express_chromo_distribution()

                count_miss, count_try = train_one_epoch(sess, ops, train_writer, data_idx=data_idx)
                validate_one_epoch(sess, ops, test_writer, epoch)
                myGA.run_one_epoch(count_miss, count_try)
                # _ = update_pred_map(sess, ops, test_writer, epoch)

                log_list(all_genome_disb_fout, count_try)
                log_list(all_genome_fitness_fout, myGA.myGenePool.get_fitness_matrix())
                log_value(used_genome_len_fout, sum(count_try))

                log_list(individual_fitness_fout, myGA.get_people_fitness_list())

            # Save the models to /dump.
            if epoch % 50 == 0:
                save_path = saver.save(sess, os.path.join(DUMP_DIR, "model.ckpt"))
                print("Model saved in file: %s" % save_path)
            # if epoch == 159:
            #     save_path = saver.save(sess, os.path.join(DUMP_DIR, "model-160.ckpt"))
            #     print("Model saved in file: %s" % save_path)
            if epoch == MAX_EPOCH-1:
                save_path = saver.save(sess, os.path.join(DUMP_DIR, "model-final.ckpt"))
                print("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer, data_idx=None):
    """ ops: dict mapping from string to tf ops """
    # sess, ops, train_writer
    is_training = True

    # if not using EA, train_data = training set
    if not EA_SCHEME:
        # not using EA, init data_idx with [0,1,2, ..., total_num_data]
        data_idx = np.arange(0, TRAIN_DATA.shape[0])

    else:
        # if using EA, data_idx was input with type of list, thus transfer it into np.array
        data_idx = np.array(data_idx)
    print('Selected training data size:', data_idx.shape[0])

    # TensorFlow can only accept batch of data
    # append random idx in dataset to fullfill the very last batch of input data to be size of BATCH_SIZE
    while data_idx.shape[0]%BATCH_SIZE!=0:
        data_idx = np.append(data_idx, nr.randint(TOTAL_GENOME))

    # size of input data in this epoch
    data_size = data_idx.shape[0]
    # print('Number of fed data:', data_size)
    # output number of input data
    FED_DATA_LEN_FOUT.write('%d\n' % (data_size))
    FED_DATA_LEN_FOUT.flush()

    log_list(FED_DATA_LIST_FOUT, data_idx)
    # number of batches in this epoch
    num_batches = data_size//BATCH_SIZE

    print('augmented training data size:', data_size)
    print('augmented training batches', num_batches)

    # Shuffle train data
    np.random.shuffle(data_idx)

    # init some outputs
    total_correct = 0
    total_miss = 0
    total_seen = 0
    loss_sum = 0

    # save the learning times of patterns & times of missclassify ones
    count_miss = np.full((TOTAL_GENOME), 0)
    count_try = np.full((TOTAL_GENOME), 0)

    total_seen_class=np.full((NUM_CLASS), 0.0)
    correct_seen_class=np.full((NUM_CLASS), 0.0)

    for batch_idx in range(num_batches):
        # set start_idx & end_idx
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        # get feed_data/label/idx for this epoch
        feed_idx = data_idx[start_idx:end_idx]
        feed_data = TRAIN_DATA[feed_idx, :, :]
        feed_label = TRAIN_LABEL[feed_idx]
        
        # Augment batched point clouds by rotation and jittering
        if ROTATE_PCL:
            if ROTATE_TYPE=='up':
                feed_data = provider.rotate_point_cloud(feed_data)
            if ROTATE_TYPE=='full':
                feed_data = provider.rotate_point_cloud_full(feed_data)
        if JITTER_PCL:
            feed_data = provider.jitter_point_cloud(feed_data, sigma=JITTER_SIGMA)

        feed_dict = {ops['pointclouds_pl']: feed_data,
                     ops['labels_pl']: feed_label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        # predicted label
        pred_label = np.argmax(pred_val, axis=1)
        # if predicted_label = true_label --> save to correct
        correct_pred = (pred_label == feed_label)
        correct = np.sum(correct_pred)
        # add-up total_correct
        total_correct += correct
        # add-up total_seen
        total_seen += BATCH_SIZE
        # calculate total miss_classify
        # total_miss = total_seen-total_correct
        # add-up cross-entropy loss
        loss_sum += loss_val
        # add-up counting for correctly classified label
        correct_class_label = [ai for ai, bi in zip(feed_label, correct_pred) if bi==1]
        
        # sum up number of all feed data with class
        for lb in feed_label:
            total_seen_class[lb] += 1
        # print('total number of eval data:', np.sum(total_seen_class))
        # sum up number of correctly classified class
        for lb in correct_class_label:
            correct_seen_class[lb] += 1

        # feed_idx & correct_pred share same fed data sequence
        # missclassify index: if correct_pred == 0 --> take the idx from feed_idx
        miss_class_idx = [ai for ai, bi in zip(feed_idx, correct_pred) if bi==0]
        
        # add up count into count_miss: record missclassified fed data
        for idx in miss_class_idx:
            count_miss[idx] += 1
        # add up count into count_try: record all fed data
        for idx in feed_idx:
            count_try[idx]+=1
        

    train_mean_loss = loss_sum / float(num_batches)
    train_ovarall_acc = total_correct / float(total_seen)
    train_avg_class_acc = np.mean(correct_seen_class/total_seen_class)

    # log out training loss
    TRAIN_LOSS_FOUT.write('%f\n' % (train_mean_loss))
    TRAIN_LOSS_FOUT.flush()

    # log out training overall acc
    TRAIN_OVERALL_ACC_FOUT.write('%f\n' % (train_ovarall_acc))
    TRAIN_OVERALL_ACC_FOUT.flush()

    # log out training average class acc
    TRAIN_AVG_CLASS_ACC_FOUT.write('%f\n' % (train_avg_class_acc))
    TRAIN_AVG_CLASS_ACC_FOUT.flush()

    # log out train log
    TRAIN_LOG_FOUT.write('%f %f\n' % (train_ovarall_acc, train_avg_class_acc))
    TRAIN_LOG_FOUT.flush()

    log_string(LOG_FOUT,'train mean loss: %f' % train_mean_loss)
    log_string(LOG_FOUT,'train overall acc: %f' % train_ovarall_acc)
    log_string(LOG_FOUT,'train avg class acc: %f' % train_avg_class_acc)
    
    class_acc_list = []
    for idx in range(NUM_CLASS):
        one_class_acc = correct_seen_class[idx] / float(total_seen_class[idx])
        class_acc_list.append(one_class_acc)
        log_string(LOG_FOUT,'train acc, correct_seen/total_seen for class no.%d: %f, %d / %d' % (idx, one_class_acc,correct_seen_class[idx], total_seen_class[idx]))
        TRAIN_EA_LOG_FOUT.write('%f %d %d ' % (one_class_acc,correct_seen_class[idx], total_seen_class[idx]))
    TRAIN_EA_LOG_FOUT.write('\n')
    TRAIN_EA_LOG_FOUT.flush()

    return count_miss, count_try



def validate_one_epoch(sess, ops, test_writer, epoch_idx):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0.0
    total_seen = 0.0
    loss_sum = 0.0

    total_seen_class=np.full((NUM_CLASS), 0.0)
    total_correct_class=np.full((NUM_CLASS), 0.0)

    # init training_data idx and shuffle the idx
    data_idx = np.arange(0, TRAIN_DATA.shape[0])

    # original evaluating datasize in this epoch
    origin_file_size = TRAIN_DATA.shape[0]
    very_last_size = origin_file_size%BATCH_SIZE
    # to add extra data for evaluating
    if very_last_size!=0:
        while data_idx.shape[0]%BATCH_SIZE!=0:
            data_idx = np.append(data_idx, 0)
    # augmented evaluating datasize
    augmented_file_size = data_idx.shape[0]
    
    # calculate num_batches for validating
    num_batches = augmented_file_size//BATCH_SIZE
    # recalculate number of validating data
    file_size = num_batches*BATCH_SIZE
    
    print('validate data size:', origin_file_size)
    print('validate data size augmented:', augmented_file_size)
    
    running_iter=num_batches-1
    if very_last_size==0:
        running_iter+=1
    
    for batch_idx in range(running_iter):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_idx = data_idx[start_idx:end_idx]
        feed_data = TRAIN_DATA[feed_idx, :, :]
        feed_label = TRAIN_LABEL[feed_idx]

        feed_dict = {ops['pointclouds_pl']: feed_data,
                     ops['labels_pl']: feed_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        
        # calculate predict label using argmax from pred_val
        pred_label = np.argmax(pred_val, axis=1)
        # if predicted_label = true_label --> save to correct
        correct_pred = (pred_label == feed_label)
        # sum up number of correctly predict
        correct = np.sum(correct_pred)
        # add-up total_correct
        total_correct += correct
        # add-up total_seen
        total_seen += BATCH_SIZE
        # calculate total miss_classify
        # total_miss = total_seen-total_correct
        # add-up cross-entropy loss
        loss_sum += loss_val

        # add-up counting for correctly classified label
        correct_class_label = [ai for ai, bi in zip(feed_label, correct_pred) if bi==1]
        # sum up number of all feed data with class
        for idx in feed_label:
            total_seen_class[idx] += 1
        # sum up number of correctly classified class
        for idx in correct_class_label:
            total_correct_class[idx] += 1


    # the very last evaluating batch
    if very_last_size!=0:
        batch_idx = num_batches-1
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_idx = data_idx[start_idx:end_idx]
        feed_data = TRAIN_DATA[feed_idx, :, :]
        feed_label = TRAIN_LABEL[feed_idx]

        feed_dict = {ops['pointclouds_pl']: feed_data,
                     ops['labels_pl']: feed_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        
        # calculate predict label using argmax from pred_val
        pred_label = np.argmax(pred_val, axis=1)
        # if predicted_label = true_label --> save to correct
        correct_pred = (pred_label[:very_last_size] == feed_label[:very_last_size])
        # sum up number of correctly predict
        correct = np.sum(correct_pred)
        # add-up total_correct
        total_correct += correct
        # add-up total_seen
        total_seen += very_last_size
        # calculate total miss_classify
        # total_miss = total_seen-total_correct
        # add-up cross-entropy loss
        loss_sum += loss_val
        # add-up counting for correctly classified label
        correct_class_label = [ai for ai, bi in zip(feed_label[:very_last_size], correct_pred) if bi==1]
    
        # sum up number of all feed data with class
        for idx in feed_label[:very_last_size]:
            total_seen_class[idx] += 1
        # print('total number of validate data:', np.sum(total_seen_class))
        # sum up number of correctly classified class
        for idx in correct_class_label:
            total_correct_class[idx] += 1

    # print('number of validate data for each clas:', total_seen_class)

    validate_mean_loss = loss_sum / float(num_batches)
    total_acc = total_correct / float(total_seen)
    avg_class_acc = np.mean(total_correct_class/total_seen_class)

    # log out validate log
    VAL_LOSS_FOUT.write('%f\n' % (validate_mean_loss))
    VAL_LOSS_FOUT.flush()
    VAL_LOG_FOUT.write('%f %f\n' % (total_acc, avg_class_acc))
    VAL_LOG_FOUT.flush()
    
    VAL_ACC_FOUT.write('%f\n' % (total_acc))
    VAL_ACC_FOUT.flush()
    VAL_AVG_CLASS_ACC_FOUT.write('%f\n' % avg_class_acc)
    VAL_AVG_CLASS_ACC_FOUT.flush()

    log_string(LOG_FOUT,'validate overall accuracy: %f'% (total_acc))
    log_string(LOG_FOUT,'validate avg class acc: %f' % (avg_class_acc))

    total_class_acc = []
    for idx in range(NUM_CLASS):
        one_class_acc = total_correct_class[idx]/float(total_seen_class[idx])
        total_class_acc.append(one_class_acc) 
        log_string(LOG_FOUT,'validate_acc, correct_seen/total_seen for class no.%d: %f, %d / %d' % (idx,one_class_acc,total_correct_class[idx], total_seen_class[idx]))
        VAL_EA_LOG_FOUT.write('%f %d %d ' % (one_class_acc,total_correct_class[idx], total_seen_class[idx]))
    VAL_EA_LOG_FOUT.write('\n')
    VAL_EA_LOG_FOUT.flush()
    log_list(VAL_CLASS_ACC_FOUT, total_class_acc)


#################################
def restore_model(model_path):
    is_training = False
    ## cpu: with tf.device('/cpu:0'):
    ## gpu: with tf.device('/gpu:'+ str(GPU_INDEX)):
    ## tpu: with tf.device('/TPU:0'):
    with tf.device('/gpu:'+ str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL_EVAL.placeholder_inputs(1, NUM_POINT)
        is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points, maxpool_idx = MODEL_EVAL.get_model(pointclouds_pl, is_training_pl)
        loss = MODEL_EVAL.get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.compat.v1.train.Saver()
        
    # Create a session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    ################################################
    ## cpu: config.log_device_placement = True
    ################################################
    ## gpu: config.gpu_options.allow_growth = True
    ##      config.allow_soft_placement = True
    ##      config.log_device_placement = True
    ################################################
    sess = tf.compat.v1.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, model_path)
    print('MODEL_PATH:',model_path)
    print("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
             'labels_pl': labels_pl,
             'is_training_pl': is_training_pl,
             'pred': pred,
             'loss': loss,
             'maxpool_idx': maxpool_idx}

    return sess, ops


def eval_test_single_dataset(save_dir, sess, ops, num_votes=1, topk=1):
    is_training = False

    total_seen_class=np.full((NUM_CLASS), 0.0)
    total_correct_class=np.full((NUM_CLASS), 0.0)

    # init testing_data idx
    file_size = TEST_LABEL.shape[0]
    data_idx = np.arange(0, file_size)

    fpoint_idx_list = []  # list of idx of featured points
    pred_label_list = np.zeros(file_size) # list of prediction
    
    for idx in range(file_size):
        start_idx = idx
        end_idx = (idx+1)

        feed_idx = data_idx[start_idx:end_idx]
        feed_data = TEST_DATA[feed_idx, :, :]
        feed_label = TEST_LABEL[feed_idx]

        feed_dict = {ops['pointclouds_pl']: feed_data,
                     ops['labels_pl']: feed_label,
                     ops['is_training_pl']: is_training}
        maxpool_idx, pred_val = sess.run([ops['maxpool_idx'], ops['pred']],
                                    feed_dict=feed_dict)

        fpoint_idx_list.append(np.reshape((maxpool_idx//1024), (1024)))
        
        # calculate predict label using argmax from pred_val
        pred_label = np.argmax(pred_val, axis=1)
        pred_label_list[idx] = pred_label

        # if predicted_label = true_label --> save to correct
        correct_pred = (pred_label == feed_label)
        # add-up counting for correctly classified label
        correct_class_label = [ai for ai, bi in zip(feed_label, correct_pred) if bi==1]

        for idx in feed_label:
            total_seen_class[idx] += 1
        # sum up number of correctly classified class
        for idx in correct_class_label:
            total_correct_class[idx] += 1


    correct_list = pred_label_list == TEST_LABEL
    overall_acc = np.sum(correct_list)/file_size
    avg_class_acc = np.mean(total_correct_class/total_seen_class)

    return fpoint_idx_list, correct_list, overall_acc, avg_class_acc


def test_single_dataset():
    # test with TEST_NAME, TEST_DATA, TEST_LABEL

    # Define model_saving dir
    MODEL_BASE_DIR = DUMP_BASE_DIR

    # Define result_saving dir
    RESULT_BASE_DIR = os.path.dirname(LOG_BASE_DIR)

    # Create save-dir
    save_dir = os.path.join(RESULT_BASE_DIR, TEST_NAME)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    overall_acc_fout = open(os.path.join(save_dir, 'overall_acc.txt'), 'w+')
    avg_class_acc_fout = open(os.path.join(save_dir, 'avg_class_acc.txt'), 'w+')

    # Load models:
    for i in range(10):
        reiter_save_dir = os.path.join(save_dir, str(i))
        if not os.path.exists(reiter_save_dir): os.makedirs(reiter_save_dir)

        model_dir = os.path.join(MODEL_BASE_DIR, str(i))
        model_dir = os.path.join(model_dir, "model-final.ckpt")

        with tf.Graph().as_default():
            sess, ops = restore_model(model_dir)

        fpoint_idx_list, correct_list, overall_acc, avg_class_acc = eval_test_single_dataset(save_dir, sess, ops)

        overall_acc_fout.write('%.6f\n' % overall_acc)
        overall_acc_fout.flush()
        
        avg_class_acc_fout.write('%.6f\n' % avg_class_acc)
        avg_class_acc_fout.flush()

        np.savetxt(os.path.join(reiter_save_dir, 'single_acc_list.txt'), correct_list, fmt='%d')
        np.savetxt(os.path.join(reiter_save_dir, 'featured_points_idx.txt'), fpoint_idx_list, fmt='%d')


def eval_test_list_dataset(test_idx, save_dir, sess, ops, num_votes=1, topk=1):
    is_training = False

    # init testing_data idx
    file_size = TEST_LABEL_LIST[test_idx].shape[0]
    data_idx = np.arange(0, file_size)

    fpoint_idx_list = []  # list of idx of featured points
    pred_label_arr = np.zeros(file_size) # list of prediction
    
    for idx in range(file_size):
        start_idx = idx
        end_idx = (idx+1)

        feed_idx = data_idx[start_idx:end_idx]
        feed_data = TEST_DATA_LIST[test_idx][feed_idx, :, :]
        feed_label = TEST_LABEL_LIST[test_idx][feed_idx]

        feed_dict = {ops['pointclouds_pl']: feed_data,
                     ops['labels_pl']: feed_label,
                     ops['is_training_pl']: is_training}
        maxpool_idx, pred_val = sess.run([ops['maxpool_idx'], ops['pred']],
                                    feed_dict=feed_dict)

        fpoint_idx_list.append(np.reshape((maxpool_idx//1024), (1024)))
        
        pred_label_arr[idx] = np.argmax(pred_val, 1)

    correct_list = pred_label_arr == TEST_LABEL_LIST[test_idx]
    overall_acc = np.sum(correct_list)/file_size

    return fpoint_idx_list, correct_list, overall_acc


def test_list_dataset():
    # test with TEST_NAME_LIST, TEST_DATA_LIST, TEST_LABEL_LIST

    # Define model_saving dir
    MODEL_BASE_DIR = DUMP_BASE_DIR

    # Define result_saving dir
    RESULT_BASE_DIR = os.path.dirname(LOG_BASE_DIR)

    for test_idx in range(len(TEST_NAME_LIST)):

        test_name = TEST_NAME_LIST[test_idx]
        
        # Create save-dir
        save_dir = os.path.join(RESULT_BASE_DIR, test_name)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        overall_acc_fout = open(os.path.join(save_dir, 'overall_acc.txt'), 'w+')
        
        # Load models:
        for i in range(10):
            reiter_save_dir = os.path.join(save_dir, str(i))
            if not os.path.exists(reiter_save_dir): os.makedirs(reiter_save_dir)

            model_dir = os.path.join(MODEL_BASE_DIR, str(i))
            model_dir = os.path.join(model_dir, "model-final.ckpt")

            with tf.Graph().as_default():
                sess, ops = restore_model(model_dir)

            fpoint_idx_list, correct_list, overall_acc = eval_test_list_dataset(test_idx, save_dir, sess, ops)

            overall_acc_fout.write('%.6f\n' % overall_acc)
            overall_acc_fout.flush()

            np.savetxt(os.path.join(reiter_save_dir, 'single_acc_list.txt'), correct_list, fmt='%d')
            np.savetxt(os.path.join(reiter_save_dir, 'featured_points_idx.txt'), fpoint_idx_list, fmt='%d')


if __name__ == "__main__":

    TOTAL_GENOME = TRAIN_DATA.shape[0]
    NUM_TEST_DATA = TEST_DATA.shape[0]

    # Save Exp time and FLAGs parms in LOG_BASE_DIR & DUMP_BASE_DIE
    LOG_FOUT = open(os.path.join(LOG_BASE_DIR, 'exp_log.txt'), 'w+')
    LOG_FOUT.write('Exp date: ' + str(date.today())+'\n')
    LOG_FOUT.write(str(FLAGS)+'\n')
    LOG_FOUT.flush()
    LOG_FOUT.close()

    LOG_FOUT = open(os.path.join(DUMP_BASE_DIR, 'exp_log.txt'), 'w+')
    LOG_FOUT.write('Exp date: ' + str(date.today())+'\n')
    LOG_FOUT.write(str(FLAGS)+'\n')
    LOG_FOUT.flush()
    LOG_FOUT.close()

    # for training re-iteration
    for EXP_IDX in range(EXP_START_IDX, REITERATION):
    # for EXP_IDX in range(6,10,1):

        # init dump dir (to save trained NN models)
        DUMP_DIR = os.path.join(DUMP_BASE_DIR, str(EXP_IDX))
        if not os.path.exists(DUMP_DIR): os.makedirs(DUMP_DIR)

        # init log dir (result)
        LOG_DIR = os.path.join(LOG_BASE_DIR, str(EXP_IDX))
        if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

        # Training log in LOG_DIR: train_loss, train_acc, eval_loss, eval_acc, etc. for each training
        LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w+')

        # Train & Validate loss fout
        TRAIN_LOSS_FOUT = open(os.path.join(LOG_DIR, 'train_loss.txt'), 'w+')
        VAL_LOSS_FOUT = open(os.path.join(LOG_DIR, 'validate_loss.txt'), 'w+')

        # Train & Validate acc fout
        # acc for all data, during train for train data
        TRAIN_OVERALL_ACC_FOUT = open(os.path.join(LOG_DIR, 'train_overall_acc.txt'), 'w+')
        TRAIN_AVG_CLASS_ACC_FOUT = open(os.path.join(LOG_DIR, 'train_avg_class_acc.txt'), 'w+')
        TRAIN_EA_LOG_FOUT = open(os.path.join(LOG_DIR, 'train_ea_log.txt'), 'w+')
        TRAIN_LOG_FOUT = open(os.path.join(LOG_DIR, 'train_log.txt'), 'w+')

        # acc for all data, after train for validating data
        VAL_ACC_FOUT = open(os.path.join(LOG_DIR, 'validate_acc.txt'), 'w+')
        # acc for each class for validating data
        VAL_AVG_CLASS_ACC_FOUT = open(os.path.join(LOG_DIR, 'validate_avg_class_acc.txt'), 'w+')
        VAL_CLASS_ACC_FOUT = open(os.path.join(LOG_DIR, 'validate_class_acc.txt'), 'w+')
        
        VAL_EA_LOG_FOUT = open(os.path.join(LOG_DIR, 'validate_ea_log.txt'), 'w+')
        VAL_LOG_FOUT = open(os.path.join(LOG_DIR, 'validate_log.txt'), 'w+')
        VAL_LOG_FOUT.write('total_acc  avg_class_acc\n')
        VAL_LOG_FOUT.flush()


        # number of data used for training in each epoch
        FED_DATA_LEN_FOUT = open(os.path.join(LOG_DIR, 'fed_data_len.txt'), 'w+')
        
        # number of data used for training in each epoch
        FED_DATA_LIST_FOUT = open(os.path.join(LOG_DIR, 'fed_data_list.txt'), 'w+')

        # Distribution of fed data: epoch/times-list
        FED_DATA_DISB_SAVE_DIR = os.path.join(LOG_DIR, 'fed_data_disb.txt')
        FED_DATA_DISB = []

        # Length of fed data: epoch/length
        EXPRESS_GENES_LEN_SAVE_DIR = os.path.join(LOG_DIR, 'express_genes_len.txt')
        EXPRESS_GENES_LEN = []

        # Index of fed data: epoch/idx-list
        EXPRESS_GENES_IDX_SAVE_DIR = os.path.join(LOG_DIR, 'express_genes_idx.txt')
        EXPRESS_GENES_IDX = []

        # Fitness of each data: epoch/fitness-list; fitness-->difficulty in learning
        GENEPOOL_FITNESS_MATRIX_SAVE_DIR = os.path.join(LOG_DIR, 'genepool_fitness_matrix.txt')
        GENEPOOL_FITNESS_MATRIX = []

        # Fitness of each individual in the population of EA: epoch/fitness-individual
        INDIVIDUAL_FITNESS_SAVE_DIR = os.path.join(LOG_DIR, 'individual_fitness.txt')

        if EA_SCHEME:
            # open/create log files
            all_genome_disb_fout = open(FED_DATA_DISB_SAVE_DIR, 'w+')
            used_genome_len_fout = open(EXPRESS_GENES_LEN_SAVE_DIR, 'w+')
            used_genome_idx_fout = open(EXPRESS_GENES_IDX_SAVE_DIR, 'w+')
            all_genome_fitness_fout = open(GENEPOOL_FITNESS_MATRIX_SAVE_DIR, 'w+')
            individual_fitness_fout = open(INDIVIDUAL_FITNESS_SAVE_DIR, 'w+')

        
        # if EA_SCHEME and EA_TYPE=='origin':
        #   POPULATION = EA_MODEL.People(TOTAL_GENOME, num_child=NUM_CHILD, num_random=NUM_RANDOM, mutation_rate=MUTATION_RATE, cross_over=CROSS_OVER, selection_scheme=SELECTION_SCHEME, mating_scheme=MATING_SCHEME)

        if EA_SCHEME and (EA_TYPE=='cannt'):
            myGA = EA_MODEL.GeneticAlgorithm(
                selection_scheme=SELECTION_SCHEME, #[rank_selection, fitness_proportionate_selection]
                replace_scheme=REPLACE_SCHEME, #[replace_with_children, replace_with_rank]

                # parms for Population
                num_people=NUM_PEOPLE, 
                num_child=NUM_CHILD, 
                
                # parms for GenePool
                total_num_genes=TOTAL_NUM_GENES, gene_pool_initiated_fitness=GENE_POOL_INITIATED_FITNESS, gene_pool_memory_len=GENE_POOL_MEMORY_LEN, 
                history_weighted_scheme=HISTORY_WEIGHTED_SCHEME, long_weight=LONG_WEIGHT, short_weight=SHORT_WEIGHT, 
                forget_scheme=FORGET_SCHEME, forget_threshold=FORGET_THRESHOLD, 
                # ******************* #
                
                # parms for Individuals
                individual_init_scheme = INDIVIDUAL_INIT_SCHEME,  #[rendom, full]
                murate_individual=MURATE_INDIVIDUAL, 
                murate_genes=MURATE_GENES
                # ******************* #
                )
        
        train()
        tf.compat.v1.reset_default_graph()
        LOG_FOUT.write('Exp date: ' + str(date.today())+'\n')
        LOG_FOUT.close()
        print("***************************************")

    # test after training
    if TEST_TYPE=='mech12':
        test_list_dataset()
    else:
        test_single_dataset()

    time_elapsed = (time.perf_counter() - time_start)
    # send_an_email(subject='Black finished the job!',info='Exp name:'+ EXP_NAME +'\nRunning time:'+ str(int(time_elapsed/60)) + ' minutes. \nThis is an automatic info from python. ')
    