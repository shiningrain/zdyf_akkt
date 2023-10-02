import os
import json
import sys
sys.path.append(os.path.dirname(__file__))

def run(params):
    os.environ["CUDA_VISIBLE_DEVICES"] = params['gpu']
    from develop import model_generate,summarize_result
    if params['search']:
        model_generate(
            block_type=params['block_type'],
            search=params['search'],
            data=params['data'],
            save_dir=params['save_dir'],#结果保存到临时中转文件夹
            epoch=params['epoch'],
            tuner=params['tuner'],
            trial=params['trial'],
            gpu=params['gpu'],
            init=params['init'],
            iter_num=params['iter_num']
            )
        
        summarize_result(json_path=params['json_path'],save_dir=params['save_dir'])#临时中转文件的结果在这里处理。
    else:
        model_generate(
            block_type=params['block_type'],
            search=params['search'],
            data=params['data'],
            save_dir=params['save_dir'],
            epoch=params['epoch'],
            param_path=params['param_path'])
        

if __name__=='__main__':
    # # 功能1：搜索模型
    # test_param_1={
    #     'block_type':'vgg',
    #     'search':True,
    #     'data':'cifar',
    #     'save_dir':'./result-v-c',# 包含所有搜索历史文件
    #     'json_path':'./result5/search_result.json',#搜索历史json文件保存的目录，可以在上一个save_dir中
    #     'epoch':5,
    #     'tuner':'dream',#'dream',
    #     'trial':10,
    #     'gpu':'1',
    #     'init':'normal',
    #     'iter_num':4,
    # }    
    
    # run(test_param_1)
    
    # 功能2：从参数生成模型
    test_param_2={
        'block_type':'vgg',
        'search':False,
        'data':'cifar',
        'save_dir':'./result_tmp1',
        'epoch':2,
        'param_path':'./param_mnist_vgg.pkl',#./param_mnist_resnet.pkl   ./param_cifar_xception.pkl
        'gpu':'1',
    }
    run(test_param_2)