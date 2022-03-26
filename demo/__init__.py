import os
from develop import model_generate
import json



def run(params):
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
        'save_dir':'./result',
        'epoch':2,
        'tuner':'greedy',
        'trial':1,
    }
    test_param_2={
        'block_type':'resnet',
        'search':False,
        'data':'mnist',
        'save_dir':'./result',
        'epoch':2,
        'param_path':'./param.pkl',
    }
    
    # run(test_param_1)
    run(test_param_2)