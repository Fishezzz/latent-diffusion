import tensorrt
import torch
import tensorflow as tf

print("TensorRT version:", tensorrt.__version__); assert tensorrt.Builder(tensorrt.Logger())

print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", tf.config.list_physical_devices('GPU'))
