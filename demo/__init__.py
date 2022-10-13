import os
import json
import sys
sys.path.append(os.path.dirname(__file__))

def run(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu']
    from develop import model_generate
    if params['search']:
        model_generate(
            block_type=params['block_type'],
            search=params['search'],
            data=params['data'],
            save_dir=params['save_dir'],
            epoch=params['epoch'],
            tuner=params['tuner'],
            trial=params['trial'])
    else:
        model_generate(
            block_type=params['block_type'],
            search=params['search'],
            data=params['data'],
            save_dir=params['save_dir'],
            epoch=params['epoch'],
            param_path=params['param_path'])
        

if __name__=='__main__':
    test_param_1={
        'block_type':'resnet',
        'search':True,
        'data':'mnist',
        'save_dir':'./result1',
        'epoch':10,
        'tuner':'greedy',#'dream',
        'trial':3,
        'gpu':'1',
    }
    test_param_2={
        'block_type':'resnet',
        'search':False,
        'data':'cifar',
        'save_dir':'./result_r',
        'epoch':2,
        'param_path':'./param_mnist_resnet.pkl',#./param_mnist_resnet.pkl   ./param_cifar_xception.pkl
        'gpu':'1',
    }
    
    
    # run(test_param_1)
    run(test_param_2)