# import onnxruntime
# import keras2onnx
# from tensorflow.keras.models import load_model
# import autokeras as ak
# import onnx

# autoKeras_model=load_model('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/best_model.h5',custom_objects=ak.CUSTOM_OBJECTS)
# onnx_model = keras2onnx.convert_keras(autoKeras_model, "autokeras", debug_mode=1)
# onnx.save_model(onnx_model, '/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/best_model.onnx')
# print(1)

# import onnx
# from onnx2pytorch import ConvertModel
# import torch

# onnx_model = onnx.load('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/best_model.onnx')
# pytorch_model = ConvertModel(onnx_model)
# torch.save(pytorch_model.state_dict(), '/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/best_model.pth')

# from onnx_pytorch import code_gen
# code_gen.gen('/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result/best_model.onnx', '/data/zxy/Projects/2020_ZhongDian/ak_test/zdyf_akkt/demo/result')
from x2paddle.convert import pytorch2paddle
pytorch2paddle(module=torch_module,
               save_dir="./pd_model",
               jit_type="trace",
               input_examples=[torch_input])