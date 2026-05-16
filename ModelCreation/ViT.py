import tensorflow as tf


class TransformBlock(tf.keras.layers.Layer):

    def __init__(self, num_heads=8, embedding_dim=256, dropout_rate=0.1):

        super().__init__()

        self.norm1 = tf.keras.layers.LayerNormalization()

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
            dropout=dropout_rate,
        )

        self.norm2 = tf.keras.layers.LayerNormalization()

        self.mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(embedding_dim * 4, activation="gelu"),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(embedding_dim),
            ]
        )

    def call(self, x, training=False):

        attention_output = self.attention(self.norm1(x), self.norm1(x))
        x = x + attention_output

        mlp_output = self.mlp(self.norm2(x), training=training)
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

        self.patch_embedding = tf.keras.layers.Conv2D(
            filters=embedding_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
        )

        self.cls_token = tf.Variable(
            tf.random.normal([1, 1, embedding_dim]), trainable=True
        )

        self.position_embedding = tf.Variable(
            tf.random.normal([1, self.num_patches + 1, embedding_dim]), trainable=True
        )
        self.num_transformer_blocks = num_transformer_blocks

        self.transformer_blocks = [
            TransformBlock(embedding_dim=embedding_dim)
            for _ in range(num_transformer_blocks)
        ]

        self.final_norm = tf.keras.layers.LayerNormalization()
        self.classifier = tf.keras.layers.Dense(num_classes)

    def _prepare_tokens(self, img):

        x = self.patch_embedding(img)

        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.embedding_dim])
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)

        x = tf.concat([cls_tokens, x], axis=1)
        x = x + self.position_embedding

        return x

    def call(self, img, training=False):
        x = self._prepare_tokens(img)

        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = self.final_norm(x)

        cls_output = x[:, 0]

        x = self.classifier(cls_output)

        return x
