
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D

from Preprocessing.Preprocessor import Preprocessor

class ViT:

    allowed_config={
        "shape": tuple,
        "batch_size": int,
        "patch_size": int
    }

    def config(self, **kwargs):
        for key, attr in kwargs.items():
            if key not in self.allowed_config:
                raise ValueError(f"Config key {key}")

            expected_type = self.allowed_config[key]

            if not isinstance(attr, expected_type):
                raise TypeError(f"{key} must be of type: {expected_type}, got {type(attr)}")

            setattr(self, key, attr)
        return self

    def prepare_embedded_vectors(self, img, stride):
        patches = Preprocessor.patchify(img, patch_size=self.patch_size, stride=stride)

        amount = patches.shape[0]
        patches = patches.reshape(amount, -1) # n amout with 768 values (if pathc size 16)

        # embedded = tf.keras.layers.Dense(256,)


    # def build(self):



