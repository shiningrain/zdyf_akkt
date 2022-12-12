import os
import json
import sys

sys.path.append(os.path.dirname(__file__))


from deepalchemy import run


if __name__ == '__main__':
    test_param_1 = {
        'gpu': 0,
        'modelname': 'resnet',
        'dataset': 'mnist',
        'epochs': 5,
        'init': 'normal',
        'iternum': 4
    }
    run(test_param_1)