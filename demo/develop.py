import os
import shutil
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist,cifar10
import argparse
import autokeras as ak
import pickle
import tensorflow.keras as keras

def mnist_load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = x_train.reshape(60000, 784)
    # x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def cifar10_load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def model_generate(
    block_type='resnet',
    search=True,
    data='mnist',
    save_dir='./result',
    epoch=2,
    tuner='greedy',
    trial=1,
    param_path='./param.pkl',
):

        
    root_path=save_dir
    tmp_dir=os.path.join(os.path.dirname(root_path),'tmp')
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(root_path)
    os.makedirs(tmp_dir)
    log_path=os.path.join(root_path,'log.pkl')
    
    if data=='mnist':
        (x_train, y_train), (x_test, y_test) = mnist_load_data()
    elif data=='cifar':
        (x_train, y_train), (x_test, y_test) = cifar10_load_data()
    else:
        (x_train, y_train), (x_test, y_test)=data #TODO: 如果是处理过的数据，需要给出数据路径或者读取方法

    # DEMO:1
    if search:
        
        # initialize the search log
        if not os.path.exists(log_path):
            log_dict={}
            log_dict['cur_trial']=-1
            log_dict['start_time']=time.time()
            log_dict['data']=data
            log_dict['tmp_dir']=tmp_dir

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
        
        input_node = ak.ImageInput()
        output_node = ak.ImageBlock(
            # Only search ResNet architectures.
            normalize=True,
            block_type=block_type,
        )(input_node)
        output_node = ak.ClassificationHead()(output_node)
        clf = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            overwrite=True, 
            max_trials=trial,
            directory=os.path.join(root_path,'image_classifier'),
            tuner=tuner,
            
        )
        clf.fit(x_train, y_train, epochs=epoch,root_path=root_path)
        
        model_path=os.path.join(root_path,'best_model.h5')
        if not os.path.exists(model_path):
            model = clf.export_model()
            model.save(model_path)
    else:
        # DEMO 2
        with open('./hypermodel.pkl', 'rb') as f:
            hm = pickle.load(f)
        with open('./hyperparam.pkl', 'rb') as f: #you need to input the parameter of the model here
            model_hyperparameter = pickle.load(f)
        with open(param_path, 'rb') as f: #you need to input the parameter of the model here
            param = pickle.load(f)
        model_hyperparameter.values=param

        model=hm.build(model_hyperparameter) #the model will be build by autokeras with this parameter 
        print(1)
        
        model.save(os.path.join(root_path,'best_model.h5'))


    print('finish')