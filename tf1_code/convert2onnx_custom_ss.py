import efficientnet.tfkeras
from tensorflow.keras.models import load_model
from tensorflow import nn
from tensorflow.keras.backend import shape
from tensorflow.keras.layers import Dropout

import tensorflow as tf
import tf2onnx
import onnxruntime as rt

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


def main():
    print('what')
    #semantic_model = keras.models.load_model(args.hdf5_file)
    h5file = '/home/mohan/git/backups/segmentation_models/examples/best_mode_model_filel.h5'
    semantic_model = load_model(h5file, custom_objects=customObjects)
    semantic_model.summary()
    print('hello here')


    spec = (tf.TensorSpec((None, 3, 320, 480), tf.float32, name="input"),)
    output_path = semantic_model.name + ".onnx"

    model_proto, _ = tf2onnx.convert.from_keras(semantic_model, input_signature=spec, opset=13, output_path=output_path)
    output_names = [n.name for n in model_proto.graph.output]
    print('onnx file generated')
    
if __name__ == "__main__":
    main()