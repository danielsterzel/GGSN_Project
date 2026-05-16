import tensorflow as tf
from tensorflow.keras.layers import Conv2D

from Preprocessing.Preprocessor import Preprocessor


class TransformBlock(tf.keras.layers.Layer):

    def __init__(self, num_heads=8, embedding_dim=256):

        super().__init__()

        self.norm1 = tf.keras.layers.LayerNormalization()

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim // num_heads
        )

        self.norm2 = tf.keras.layers.LayerNormalization()

        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(embedding_dim * 4, activation="relu"),
                tf.keras.layers.Dense(embedding_dim),
            ]
        )

    def call(self, x):

        attention_output = self.attention(self.norm1(x), self.norm1(x))
        x = x + attention_output

        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output

        return x


class ViT(tf.keras.Model):
    def __init__(
        self,
        num_classes,
        image_size=256,
        patch_size=16,
        embedding_dim=256,
        num_transformer_blocks=4,
    ):

        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = tf.keras.layers.Dense(embedding_dim)

        self.position_embedding = tf.Variable(
            tf.random.normal([1, self.num_patches, embedding_dim]), trainable=True
        )

        self.num_transformer_blocks = num_transformer_blocks
        self.transformer_blocks = [
            TransformBlock(embedding_dim=embedding_dim)
            for _ in range(num_transformer_blocks)
        ]

        self.classifier = tf.keras.layers.Dense(num_classes)

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

        x = self._prepare_embedded_vectors(img)

        for block in self.transformer_blocks:
            x = block(x)

        x = tf.reduce_mean(x, axis=1)
        x = self.classifier(x)

        return x

        return x
