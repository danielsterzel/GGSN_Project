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


def _collapse_repeating_text(text, min_pattern_length=1, max_pattern_length=5, min_repetitions=2):
    if len(text) < 2 * min_repetitions:
        return text

    max_pattern_length = min(max_pattern_length, len(text) // 2)

    for pattern_length in range(min_pattern_length, max_pattern_length + 1):
        pattern = text[:pattern_length]
        repetitions = 0
        index = 0

        while text.startswith(pattern, index):
            repetitions += 1
            index += pattern_length

        if repetitions >= min_repetitions and index / max(1, len(text)) >= 0.8:
            return pattern

    return text


def _should_suppress_text(original_text, filtered_text):
    if not filtered_text:
        return True

    if original_text == filtered_text:
        return False

    original_length = len(original_text)
    filtered_length = len(filtered_text)

    if original_length >= 8 and filtered_length <= 2:
        return True

    if original_length >= 12 and filtered_length <= 3:
        return True

    return False


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
    logits = tf.convert_to_tensor(logits)
    token_ids = tf.argmax(logits, axis=-1, output_type=tf.int32).numpy()

    decoded_texts = []
    for row in token_ids:
        collapsed = []
        previous_token = None

        for token_id in row:
            token_id = int(token_id)

            if token_id == vocabulary.blank_id:
                previous_token = token_id
                continue

            if previous_token == token_id:
                continue

            collapsed.append(token_id)
            previous_token = token_id

        decoded_text = vocabulary.decode(collapsed)
        filtered_text = _collapse_repeating_text(decoded_text)
        if _should_suppress_text(decoded_text, filtered_text):
            decoded_texts.append("")
        else:
            decoded_texts.append(filtered_text)

    return decoded_texts
