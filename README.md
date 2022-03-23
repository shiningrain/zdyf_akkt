<!--
 * @Author: your name
 * @Date: 2022-03-23 22:22:32
 * @LastEditTime: 2022-03-23 22:30:53
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /ak_test/zdyf_akkt/README.md
-->
# zdyf_akkt
Autokeras and Kerastuner of our project


## Usage

Replace the `/data/zxy/anaconda3/envs/ak_2.3/bin/python` in [here](./autokeras/engine/tuner.py) and [here](./autokeras/engine/tuner.py).
And then copy the `autokeras` and  `kerastuner` to replace your autokeras (1.0.12) lib and kerastuner (1.0.2) lib

```bash
$ cd zdyf_akkt
$ python demo_0.py
```

You can change the `--tuner` in demo_0.py to assign the tuner in search. The usage of 'dream' tuner refers to [this](https://github.com/shiningrain/DREAM).