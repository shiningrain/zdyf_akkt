import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
sys.path.append('./utils')
from load_test_utils import traversalDir_FirstDir
from utils_data import *
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import pandas
import tensorflow.keras as keras
import sys
import pickle
import tensorflow as tf
import time
import argparse
import autokeras as ak
import shutil
from .Deepalchemy import deepalchemy as da


IMAGE_SIZE=300# food101
#360×240 # cars196

def find_origin(root_dir):
    dir_list=traversalDir_FirstDir(root_dir)
    for trial_dir in dir_list:
        if os.path.basename(trial_dir).startswith('0-'):
            return trial_dir

def format_example(image, label,image_size=IMAGE_SIZE):
    # image_size=256
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, (image_size, image_size))#(360,240))#(image_size, image_size))#
    return image, label

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test image classification')
    parser.add_argument('--data','-d',default='mnist',choices=['cifar','mnist','cars','cifar100','food','cars','tiny'], help='dataset')
    parser.add_argument('--result_root_path','-rrp',default='./Test_dir/demo_result', help='the directory to save results')
    parser.add_argument('--tmp_dir','-td',default='./Test_dir/tmp0', help='the directory to save results')    
    parser.add_argument('--epoch','-ep',default=1, help='training epoch')
    parser.add_argument('--trials','-tr',default=5, help='searching trials')
    parser.add_argument('--tuner','-tn',default='dream',choices=['greedy','dream','bayesian','hyperband','deepalchemy'], help='searching trials')
    
    args = parser.parse_args()

    # set the path and generate the directory
    root_path=args.result_root_path
    tmp_dir=args.tmp_dir
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(root_path)
    os.makedirs(tmp_dir)
    log_path=os.path.join(root_path,'log.pkl')
 
    # load datasets
    if args.data=='cifar100':
        (x_train, y_train), (x_test, y_test)=cifar100_load_data()
    elif args.data=='cifar':
        (x_train, y_train), (x_test, y_test)=cifar10_load_data()
    elif args.data=='mnist':
        (x_train, y_train), (x_test, y_test)=mnist_load_data()

    # initialize the search log
    if not os.path.exists(log_path):
        log_dict={}
        log_dict['cur_trial']=-1
        log_dict['start_time']=time.time()
        log_dict['data']=args.data
        log_dict['tmp_dir']=args.tmp_dir

        with open(log_path, 'wb') as f:
            pickle.dump(log_dict, f)

    else:
        with open(log_path, 'rb') as f:
            log_dict = pickle.load(f)
        for key in log_dict.keys():
            if key.startswith('{}-'.format(log_dict['cur_trial'])):
                log_dict['start_time']=time.time()-log_dict[key]['time']
                break
        with open(log_path, 'wb') as f:
            pickle.dump(log_dict, f)

    # search models, if you have finished the `setup` in readme.md, the greedy
    # method is our feedback-based search method.

    if args.epoch!=None:
        epoch=int(args.epoch)
    else:
        epoch=args.epoch


    if args.tuner != 'deepalchemy':
        clf = ak.ImageClassifier(
        overwrite=True,directory=os.path.join(args.result_root_path,'image_classifier'),
        max_trials=args.trials,tuner=args.tuner)#,tuner='bayesian'

        # load the tfds datasets and feed the image classifier with training data.
        clf.fit(x_train, y_train, epochs=epoch,root_path=root_path)

    else:
        trainfunc, nmax = da.gen_train_function(False, [x_train, y_train, x_test, y_test], False, '0', epoch)
        wmin, wmax, dmin, dmax = da.NM_search_min(trainfunc, nmax, 'normal', 4)

        trainfunc, nmax = da.gen_train_function(True, [x_train, y_train, x_test, y_test], False, '0', epoch)
        valloss = trainfunc(dmin, dmax, wmin, wmax)


    print('finish')