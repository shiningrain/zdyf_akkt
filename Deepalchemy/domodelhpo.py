
from ..Deepalchemy.myModel import resnet18
import tensorflow as tf
from hyperopt import Trials, STATUS_OK, tpe
from hyperas.distributions import choice, uniform
from hyperas import optim
from ..Deepalchemy.new_evaluation import build_resnet_dicts
import numpy as np
import os
from tensorflow.keras import datasets
from tensorflow.keras.datasets import cifar10
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def create_model(deep, width):
    import numpy as np
    deep = {{uniform(dmin, dmax)}}
    deep = np.round(deep / 2) * 2
    width = {{uniform(wmin, wmax)}}
    width = np.round(width * 4) / 4
    model = resnet18(width, build_resnet_dicts()[deep])
    batch_size = {{choice([64, 128, 256, 512])}}
    
    learning_rate = {{choice([1e-2, 1e-3, 1e-4])}}
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                                                     patience=5, min_lr=1e-5)

    opname = {{choice(['adam', 'sgd'])}}
    if opname == 'adam':
        op = tf.keras.optimizers.Adam(learning_rate)
    else:
        op = tf.keras.optimizers.SGD(learning_rate)
    model.compile(optimizer=op,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['sparse_categorical_accuracy'])
    ear = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

    history = model.fit(x_train, y_train, batch_size,
                            epochs=epochs, validation_data=(x_test, y_test), validation_freq=1
                            , callbacks=[reduce_lr], verbose=0
                            )

        #del(imgGen)
    name = model.name + '_bs_' + str(batch_size) + '_lr_' + str(learning_rate) + '_epo_' + str(epochs) + '_' + opname
    val_loss = history.history['val_loss']
    model.save("./models/" + name + ".h5")
    del(model)
    #del(x_train)
    #del(y_train)
    #del(x_test)
    #del(y_test)
    np.save('./data/' + name + '_valloss', val_loss)
    return {'loss': val_loss[-1], 'status': STATUS_OK, 'model': name}


def data():
    from ..Deepalchemy import tempparas
    x_train, y_train, x_test, y_test = tempparas.x_train, tempparas.y_train, tempparas.x_test, tempparas.y_test
    epochs = tempparas.epochs
    dmin = tempparas.dmin
    dmax = tempparas.dmax
    wmin = tempparas.wmin
    wmax = tempparas.wmax   
    return x_train, y_train, x_test, y_test


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu',required=True)
    parser.add_argument('--times',required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = parse_args()
    g = args.gpu
    t = args.times
    os.environ["CUDA_VISIBLE_DEVICES"] = str(g)
    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists('./model'):
        os.mkdir('./model')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    result, name, space = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=int(t),
                                          trials=Trials(),
                                          verbose=False,
                                          return_space=True)
    valloss = np.load('./data/'+str(name)+ '_valloss.npy')[-1]
    model = tf.keras.models.load_model('./data/'+str(name)+'.h5')
    model.save("./best.h5")
    np.save('./data/best.npy',valloss)

    #print('The best model:')
    #savedict = {}
    #for i, hp in enumerate(result):
    #    print(str(hp) + ':' + str(space[hp].pos_args[result[hp] + 1].obj))
    #    savedict[hp] = space[hp].pos_args[result[hp] + 1].obj
    
    