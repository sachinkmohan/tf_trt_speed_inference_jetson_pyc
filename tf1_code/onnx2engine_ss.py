import engine as eng
from onnx import ModelProto
import tensorrt as trt

import engine_ops as eop

engine_path = './'
onnx_path = "model_1.onnx"
batch_size = 1

def main():
  eop.save_engine(engine_path, onnx_path)

if __name__ == '__main__':
  main()
