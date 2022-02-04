import keras
from keras2onnx import convert_keras
import argparse

import os
os.environ['TF_KERAS'] = '1'

import efficientnet.tfkeras
from tensorflow.keras.models import load_model

from tensorflow import nn
from tensorflow.keras.backend import shape
from tensorflow.keras.layers import Dropout

import segmentation_models as sm

class FixedDropout(Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        return tuple([shape(inputs)[i] if sh is None else sh for i, sh in enumerate(self.noise_shape)])

customObjects = {
    'swish': nn.swish,
    'FixedDropout': FixedDropout,
    'dice_loss_plus_1binary_focal_loss': sm.losses.binary_focal_dice_loss,
    'iou_score': sm.metrics.iou_score,
    'f1-score': sm.metrics.f1_score
}


def keras_to_onnx(model, output_filename):
   onnx = convert_keras(model, output_filename)
   with open(output_filename, "wb") as f:
       f.write(onnx.SerializeToString())


def main(args):
    #semantic_model = keras.models.load_model(args.hdf5_file)
    semantic_model = load_model(args.hdf5_file, custom_objects=customObjects)
    keras_to_onnx(semantic_model, args.onnx_file) 
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5_file', type=str)
    parser.add_argument('--onnx_file', type=str, default='semantic_segmentation.onnx')
    args=parser.parse_args()
    main(args)
