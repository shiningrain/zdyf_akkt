import sys
from ..Deepalchemy import new_evaluation as eva
import tensorflow as tf
import numpy as np
#import myModel
from ..Deepalchemy import myModel
import time
import os
import argparse
import pickle



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def get_model_n_params(model):
    return np.sum([tf.keras.backend.count_params(w) for w in (model.trainable_weights + model.non_trainable_weights)])


def get_resnet18_n_params(width, deep):
    resnet_dict = eva.build_resnet_dicts()
    model = myModel.resnet18(width, resnet_dict[deep])
    n = get_model_n_params(model)
    del model
    return n


def nparams_to_width(n, wmin, wmax, deep):
    if deep is None:
        deep = 18
    l = wmin
    r = wmax
    cnt_width = (l + r) / 2
    while np.abs(np.round(l * 2) / 2 - np.round(r * 2) / 2) > 0.5:
        cnt_n = get_resnet18_n_params(cnt_width, deep)
        if cnt_n > n:
            r = cnt_width
        else:
            l = cnt_width
        cnt_width = (l + r) / 2

    return np.round(cnt_width * 2) / 2


def stop_OK(current_list, result_dict, nmax):
    cnt = 0
    n_list = [0, 0, 0]
    deep_list = [0, 0, 0]
    width_list = [0, 0, 0]
    for i, j in current_list:
        if result_dict.get((i, j)) is None:
            return False
        n_list[cnt] = get_resnet18_n_params(j, i)
        deep_list[cnt] = i
        width_list[cnt] = j
        cnt = cnt + 1
    if (max(deep_list) - min(deep_list) < 3) and (max(width_list) - min(width_list) < 3):
        return True
    else:
        return False


def bound_examination(nmax, bound_dict, width, deep):
    if deep < 2:
        deep = 2
    elif deep >= 50:
        deep = 50
    if bound_dict.get(deep) is None:
        bound_dict[deep] = nparams_to_width(nmax, 0, 64, deep)
    if width < 1:
        width = 1
    elif width > bound_dict[deep]:
        width = bound_dict[deep]

    return bound_dict, width, deep


def NM_search_min(trainfunc, nmax, init_method, iternum):
    result_dict = {}
    history = []
    if init_method == 'rand':
        wrand1 = np.random.rand()
        wrand2 = np.random.rand()
        wrand3 = np.random.rand()
        drand1 = np.random.randint(1,25) *2
        drand2 = np.random.randint(1,25) *2
        drand3 = np.random.randint(1,25) *2
        # 初始三角形选定深度为18,18,34 宽度为nmax/3,2nmax/3,nmax/2对应的宽度
        current_list = [
            [drand1, nparams_to_width(nmax * wrand1, 0, 64, drand1)],
            [drand2, nparams_to_width(wrand2 * nmax, 0, 64, drand2)],
            [drand3, nparams_to_width(nmax * wrand3, 0, 64, drand3)],
        ]
        bound_dict = {
            drand1: nparams_to_width(nmax, 0, 64, drand1),
            drand2: nparams_to_width(nmax, 0, 64, drand2),
            drand3: nparams_to_width(nmax, 0, 64, drand3),
        }
    elif init_method == 'large':
        bound_dict = {
        8: nparams_to_width(nmax, 0, 64, 8),
        42: nparams_to_width(nmax, 0, 64, 42)
        }
        # 初始三角形选定深度为18,18,50 宽度为nmax/3,2nmax/3,nmax/2对应的宽度
        current_list = [
            [8, nparams_to_width(nmax / 6, 0, 64, 8)],
            [8, nparams_to_width(5 * nmax / 6, 0, 64, 8)],
            [42, nparams_to_width(nmax / 2, 0, 64, 42)],
        ]
    elif init_method == 'normal':
        bound_dict = {
        18: nparams_to_width(nmax, 0, 64, 18),
        34: nparams_to_width(nmax, 0, 64, 34)
        }
        # 初始三角形选定深度为18,18,34 宽度为nmax/3,2nmax/3,nmax/2对应的宽度
        current_list = [
            [18, nparams_to_width(nmax / 3, 0, 64, 18)],
            [18, nparams_to_width(2 * nmax / 3, 0, 64, 18)],
            [34, nparams_to_width(nmax / 2, 0, 64, 34)],
        ]
    iter = 0
    print('iter ' + str(iter))
    print('Current Singular:')
    for dp, wth in current_list:
        result_dict[(dp, wth)] = trainfunc(wth, dp)
        print('Deep = ' + str(dp) + ' Width = ' + str(wth) + ' Loss = ' + str(result_dict[(dp, wth)]))

    #while not stop_OK(current_list, result_dict, nmax):
    for times in range(iternum):
        iter = iter + 1
        print('iter ' + str(iter))
        current_list = sorted(current_list, key=lambda x: result_dict[tuple(x)])
        history.append([current_list[0],result_dict[tuple(current_list[0])]])
        deep_o = round((current_list[0][0] + current_list[1][0]) / 4) * 2
        width_o = round(current_list[0][1] + current_list[1][1]) / 2
        # result_dict[(deep_o, width_o)] = trainfunc(width_o, deep_o)

        deep_r = 2 * deep_o - current_list[2][0]
        deep_r = round(deep_r / 2) * 2
        width_r = 2 * width_o - current_list[2][1]
        bound_dict, width_r, deep_r = bound_examination(nmax, bound_dict, width_r, deep_r)
        r = trainfunc(width_r, deep_r) if result_dict.get((deep_r, width_r)) is None else result_dict[(deep_r, width_r)]
        result_dict[(deep_r, width_r)] = r

        if result_dict[tuple(current_list[1])] > r >= result_dict[tuple(current_list[0])]:
            print('Reflection')
            current_list[2] = [deep_r, width_r]
        elif r < result_dict[tuple(current_list[0])]:
            print('Expansion')
            deep_e = deep_o + 2 * (deep_r - deep_o)
            width_e = width_o + 2 * (width_r - width_o)
            bound_dict, width_e, deep_e = bound_examination(nmax, bound_dict, width_e, deep_e)
            e = trainfunc(width_e, deep_e) if result_dict.get((deep_e, width_e)) is None else result_dict[(deep_e, width_e)]
            result_dict[(deep_e, width_e)] = e
            if e < r:
                current_list[2] = [deep_e, width_e]
            else:
                current_list[2] = [deep_r, width_r]
        elif r >= result_dict[tuple(current_list[1])]:
            deep_c = deep_o + 0.5 * (current_list[2][0] - deep_o)
            deep_c = round(deep_c / 2) * 2
            width_c = width_o + 0.5 * (current_list[2][1] - width_o)
            bound_dict, width_c, deep_c = bound_examination(nmax, bound_dict, width_c, deep_c)
            c = trainfunc(width_c, deep_c) if result_dict.get((deep_c, width_c)) is None else result_dict[(deep_c, width_c)]
            result_dict[(deep_c, width_c)] = c
            if c < result_dict[tuple(current_list[2])]:
                print('Contraction')
                current_list[2] = [deep_c, width_c]
            else:
                print('Shrink')
                current_list[1][0] = current_list[0][0] + 0.5 * (current_list[1][0] - current_list[0][0])
                current_list[1][0] = round(current_list[1][0] / 2) * 2
                current_list[1][1] = current_list[0][1] + 0.5 * (current_list[1][1] - current_list[0][1])
                current_list[2][0] = current_list[0][0] + 0.5 * (current_list[2][0] - current_list[0][0])
                current_list[2][0] = round(current_list[2][0] / 2) * 2
                current_list[2][1] = current_list[0][1] + 0.5 * (current_list[2][1] - current_list[0][1])
                for dp, wth in current_list:
                    if result_dict.get((dp, wth)) is None:
                        result_dict[(dp, wth)] = trainfunc(wth, dp) 
        print('Current Singular:')
        for dp, wth in current_list:
            print('Deep = ' + str(dp) + ' Width = ' + str(wth) + ' Loss = ' + str(result_dict[(dp, wth)]))
    print(history)
    current_list = sorted(current_list, key=lambda x: result_dict[tuple(x)])
    return min([k[0] for k in current_list]), max([k[0] for k in current_list]), min([k[1] for k in current_list]), max([k[1] for k in current_list])
    #return current_list[0], result_dict[tuple(current_list[0])]


def calc_nmax(n_dataset, imggen_dict):
    n_max = n_dataset
    return n_max


def write_temp(key, data, epochs, **kwargs):
    with open('tempparas.py','w') as f:
        f.write('import tensorflow as tf\n')
        f.write('import datasets\n')
        f.write('import new_evaluation as eva\n')
        f.write('from ..demo.utils.utils_data import *\n')

        if key == 0:
            f.write('dmin = '+str(kwargs['dmin'])+'\n')
            f.write('dmax = '+str(kwargs['dmax'])+'\n')
            f.write('wmin = '+str(kwargs['wmin'])+'\n')
            f.write('wmax = '+str(kwargs['wmax'])+'\n')
        else:
            f.write('d = '+str(kwargs['d'])+'\n')
            f.write('w = '+str(kwargs['w'])+'\n')    
        if data == 'cifar':
            f.write('x_train, y_train, x_test, y_test = cifar10_load_data()\n')
        elif data == 'mnist':
            f.write('x_train, y_train, x_test, y_test = mnist_load_data()\n')

        f.write('epochs = '+str(epochs)+'\n')

        f.close()


def gen_train_function(hpo, dataset, crop, gpu, epochs,data):
    x_train, y_train, x_test, y_test = dataset

    nmax = y_train.shape[0]
    use_imggen = 0
    d0 = 18
    w0 = nparams_to_width(nmax, 0, 64, 18)

    write_temp(1, data, epochs, d=d0, w=w0)
    os.system('python center.py --gpu='+gpu+' --times=5')
    cdict = np.load('./center.npy',allow_pickle=True)[()]
    bs = cdict['batch_size']
    lr = cdict['learning_rate']
    opname = cdict['opname']
    def trainfunc(width, deep):
        if deep is None:
            deep = 18
        model = myModel.resnet18(width, eva.build_resnet_dicts()[deep])
        model, acc, vacc, loss, vloss = eva.train_with_hypers(model, x_train, y_train, x_test, y_test, bs, lr, epochs, opname)
        return vloss[-1]
    
    def trainfunc_hpo(dmin, dmax, wmin, wmax):

        write_temp(0, data, epochs, dmin=dmin, dmax=dmax, wmin=wmin, wmax=wmax)
        os.system('python domodelhpo.py --gpu='+gpu+' --times=5')
        valloss = np.load('./data/best.npy')
        return valloss
    return trainfunc if not hpo else trainfunc_hpo, nmax




