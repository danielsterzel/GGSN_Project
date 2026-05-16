import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D

from Preprocessing.Preprocessor import Preprocessor


class ViT(tf.keras.Model):
    def __init__(self, image_size=256, patch_size=16, embedding_dim=256):

        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = tf.keras.layers.Dense(embedding_dim)

        self.position_embedding = tf.Variable(
            tf.random.normal([1, self.num_patches, embedding_dim]), trainable=True
        )
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=self.embedding_dim
        )

    def _prepare_embedded_vectors(self, img):
        patches = Preprocessor.patchify(
            img, patch_size=self.patch_size, stride=self.patch_size
        )

        patches = patches.reshape(self.num_patches, -1)

        embedded = self.patch_embedding(patches)

        embedded = tf.expand_dims(embedded, axis=0)

        embedded = embedded + self.position_embedding

        return embedded

    def call(self, img):

        embedded = self._prepare_embedded_vectors(img)
        attention_output = self.attention(embedded, embedded)

        return attention_output
