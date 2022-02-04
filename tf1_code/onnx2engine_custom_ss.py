import efficientnet.tfkeras

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import nn
from tensorflow.keras.backend import shape
from tensorflow.keras.layers import Dropout

import tf2onnx
import onnxruntime as rt

import segmentation_models as sm

import keras

#from tensorflow.python.keras._impl import keras

from keras2onnx import convert_keras
from engine import *


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


def main():
    print('what')
    # semantic_model = keras.models.load_model(args.hdf5_file)
    h5file = '/home/mohan/git/backups/segmentation_models/examples/best_mode_model_filel.h5'
    model = load_model(h5file, custom_objects=customObjects)
    print('hello here')

    onnx_path = 'model_1.onnx'
    engine_name = 'model_1.plan'

    batch_size = 1
    CHANNEL = 3
    HEIGHT = 320
    WIDTH = 480


    model._layers[0].batch_input_shape = (None, 320, 480,3)
    model = keras.models.clone_model(model)

    onx = convert_keras(model, onnx_path)
    with open(onnx_path, "wb") as f:
        f.write(onx.SerializeToString())

    shape = [batch_size, HEIGHT, WIDTH, CHANNEL]
    engine = build_engine(onnx_path, shape=shape)
    save_engine(engine, engine_name)
    print('done')

if __name__ == "__main__":
    main()