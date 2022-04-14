# zdyf_akkt
Autokeras and Kerastuner of our project


## Usage

Use your python path to replace the `/data/zxy/anaconda3/envs/ak_2.3/bin/python` in [here](./Autokeras/engine/tuner.py) and [here](./utils/load_test_utils.py).
And then copy the `autokeras` and  `kerastuner` to replace your autokeras (1.0.12) lib and kerastuner (1.0.2) lib

```bash
$ cd zdyf_akkt
$ python demo_0.py
```

You can change the `--tuner` in demo_0.py to assign the tuner in search. The usage of 'dream' tuner refers to [this](https://github.com/shiningrain/DREAM).


## TODO
- [ ] 加入更多的模型支持（目前支持VGG、NasNet、DenseNet、EfficientNet、ResNet、CNN、Xception，总计7种网络架构）
  - [x]  VGG
  - [x]  NasNet
  - [x]  DenseNet
  - [ ]  ViT (transformer)——只有最新版本TensorFlow可以用transformer网络，目前考虑后续兼容框架测试使用的是2.3的TensorFlow
- [ ] 加入更多超参数用于自动化生成模型，拓展可搜索模型空间
  - [ ] Activation
  - [ ] Initializer
  - [x] Optimizer
- [x] 构建简单的上层api
  - [x] 整理Demo （两种模式，自动搜索\仅构建模型）
  - [x] 训练记录与对应输出接口
- [x] 将我们已有的模型搜索方法集成进去
  - [x] DREAM search
