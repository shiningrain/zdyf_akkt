'''
Author: your name
Date: 2022-03-19 20:43:15
LastEditTime: 2022-03-19 21:32:25
LastEditors: Please set LastEditors
Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
FilePath: /ak_test/demo1.py
'''
import os
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if data=='mnist':
        (x_train, y_train), (x_test, y_test) = mnist_load_data()
    elif data=='cifar':
        (x_train, y_train), (x_test, y_test) = cifar10_load_data()
    else:
        (x_train, y_train), (x_test, y_test)=data #TODO: 如果是处理过的数据，需要给出数据路径或者读取方法

    # DEMO:1
    if search:
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
            directory=save_dir,
            tuner=tuner,
            
        )
        clf.fit(x_train, y_train, epochs=epoch)

        model = clf.export_model()
        
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
        
    model.save(os.path.join(save_dir,'model-demo1.h5'))


    # loaded_model = load_model(os.path.join(args.save_dir,'model.h5'), custom_objects=ak.CUSTOM_OBJECTS)
    # predicted_y = loaded_model.predict(tf.expand_dims(x_test))
    # print(predicted_y)
    print('finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo, search model or build model')
    parser.add_argument('-block_type','-blk',default='resnet',choices=['efficientnet','resnet','xception','valinia'], help='model architecture') # TODO: VGG,nasnet,lenet
    parser.add_argument('-search','-s',default=True, help='True will search model, False will build model from parameter')
    parser.add_argument('-data','-d',default='mnist',choices=['cifar','mnist'], help='dataset')
    parser.add_argument('-save_dir','-sd',default='./result', help='model save directory')
    parser.add_argument('-epoch','-ep',default=2, help='maximum training epoch')
    parser.add_argument('-tuner','-tn',default='greedy', help='ONLY when -s=True, search tuner')
    parser.add_argument('-trial','-tr',default=1, help='ONLY when -s=True, search trial')
    parser.add_argument('-param_path','-pp',default='./param.pkl', help='ONLY when -s=False, the model parameter path, a dict saved as pickle')
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if args.data=='mnist':
        (x_train, y_train), (x_test, y_test) = mnist_load_data()
    elif args.data=='cifar':
        (x_train, y_train), (x_test, y_test) = cifar10_load_data()

    # DEMO:1
    if args.search:
        input_node = ak.ImageInput()
        output_node = ak.ImageBlock(
            # Only search ResNet architectures.
            block_type=args.block_type,
        )(input_node)
        output_node = ak.ClassificationHead()(output_node)
        clf = ak.AutoModel(
            inputs=input_node,
            outputs=output_node,
            overwrite=True, 
            max_trials=args.trial,
            directory=args.save_dir,
            tuner=args.tuner,
            
        )
        clf.fit(x_train, y_train, epochs=args.epoch)

        model = clf.export_model()
        
    else:
        # DEMO 2
        with open('./hypermodel.pkl', 'rb') as f:
            hm = pickle.load(f)
        with open('./hyperparam.pkl', 'rb') as f: #you need to input the parameter of the model here
            model_hyperparameter = pickle.load(f)
        with open(args.param_path, 'rb') as f: #you need to input the parameter of the model here
            param = pickle.load(f)
        model_hyperparameter.values=param

        model=hm.build(model_hyperparameter) #the model will be build by autokeras with this parameter 
        print(1)
        
    model.save(os.path.join(args.save_dir,'model-demo1.h5'))


    # loaded_model = load_model(os.path.join(args.save_dir,'model.h5'), custom_objects=ak.CUSTOM_OBJECTS)
    # predicted_y = loaded_model.predict(tf.expand_dims(x_test))
    # print(predicted_y)
    print('finish')