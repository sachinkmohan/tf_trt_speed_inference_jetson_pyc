import engine as eng
import argparse
from onnx import ModelProto 
#import tensorrt as trt
import tensorflow.contrib.tensorrt as trt
 
 
def main():
    engine_name = 'ss_0202.plan'
    onnx_path = './ss_0202.onnx'
    batch_size = 1 
    
    model = ModelProto()
    with open(onnx_path, "rb") as f:
        model.ParseFromString(f.read())
    
    a1 = model.graph.input
    #a1_dim = a1.type.tensor_type.shape.dim
    #print('a1', a1)
    #print('a1_dim', a1_dim)
    #d0 = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
    #print('printing d0 ->', d0)
    #d1 = model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    #d2 = model.graph.input[0].type.tensor_type.shape.dim[3].dim_value
    #shape = [batch_size , d0, d1 ,d2]
    shape = [1,1,320, 480, 3]
    engine = eng.build_engine(onnx_path, shape= shape)
    eng.save_engine(engine, engine_name) 
 
 
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--onnx_file', type=str)
    # parser.add_argument('--plan_file', type=str, default='engine.plan')
    # args=parser.parse_args()
    main()
