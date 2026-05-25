import tensorflow as tf
import string


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


class ViTEncoder(tf.keras.Model):
    def __init__(
        self,
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

    def _prepare_tokens(self, img):

        x = self.patch_embedding(img)

        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, [batch_size, -1, self.embedding_dim])
        cls_tokens = tf.repeat(self.cls_token, repeats=batch_size, axis=0)

        x = tf.concat([cls_tokens, x], axis=1)
        x = x + self.position_embedding

        return x

    def call(self, img, training=False, return_sequence=False):
        x = self._prepare_tokens(img)

        for block in self.transformer_blocks:
            x = block(x, training=training)

        x = self.final_norm(x)

        if return_sequence:
            return x

        cls_output = x[:, 0]

        return cls_output



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

        self.encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            num_transformer_blocks=num_transformer_blocks,
        )
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, img, training=False):
        cls_output = self.encoder(img, training=training)

        return self.classifier(cls_output)


class ViTOCR(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        image_size=256,
        patch_size=16,
        embedding_dim=256,
        num_transformer_blocks=4,
    ):

        super().__init__()

        self.encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            num_transformer_blocks=num_transformer_blocks,
        )
        self.token_projection = tf.keras.layers.Dense(vocab_size)

    def call(self, img, training=False):
        sequence = self.encoder(img, training=training, return_sequence=True)

        patch_tokens = sequence[:, 1:, :]
        logits = self.token_projection(patch_tokens)

        return logits


class OCRVocabulary:
    def __init__(self, characters=None):
        if characters is None:
            characters = string.ascii_lowercase + string.ascii_uppercase + string.digits

        self.characters = characters
        self.blank_id = 0
        self.char_to_id = {char: index + 1 for index, char in enumerate(characters)}
        self.id_to_char = {index + 1: char for index, char in enumerate(characters)}

    @property
    def size(self):
        return len(self.characters) + 1

    def encode(self, text):
        return [self.char_to_id[char] for char in text if char in self.char_to_id]

    def decode(self, token_ids):
        decoded_chars = []

        for token_id in token_ids:
            token_id = int(token_id)

            if token_id == self.blank_id:
                continue

            char = self.id_to_char.get(token_id)
            if char is not None:
                decoded_chars.append(char)

        return "".join(decoded_chars)


def build_ctc_input_lengths(batch_size, time_steps):
    return tf.fill([batch_size, 1], tf.cast(time_steps, tf.int32))


def build_ctc_label_lengths(labels):
    return tf.math.count_nonzero(labels, axis=1, keepdims=True, dtype=tf.int32)


def ctc_loss(labels, logits):
    batch_size = tf.shape(logits)[0]
    time_steps = tf.shape(logits)[1]

    input_lengths = build_ctc_input_lengths(batch_size, time_steps)
    label_lengths = build_ctc_label_lengths(labels)

    y_pred = tf.nn.softmax(logits, axis=-1)

    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_lengths, label_lengths)


def greedy_ctc_decode(logits, vocabulary):
    input_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
    probabilities = tf.nn.softmax(logits, axis=-1)
    decoded, _ = tf.keras.backend.ctc_decode(
        probabilities, input_length=input_length, greedy=True
    )

    token_ids = decoded[0].numpy()

    return [vocabulary.decode(row) for row in token_ids]
