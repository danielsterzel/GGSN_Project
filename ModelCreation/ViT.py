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
        # create trainable positional and cls token weights in build(), so shapes
        # can depend on the actual input image size when used in a model.
        self.cls_token = None
        self.position_embedding = None
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

    def build(self, input_shape):
        # input_shape: (batch, height, width, channels)
        try:
            height = int(input_shape[1])
            width = int(input_shape[2])
        except Exception:
            # fall back to configured image size
            height = self.image_size
            width = self.image_size

        patches_h = max(1, height // self.patch_size)
        patches_w = max(1, width // self.patch_size)
        self.num_patches = patches_h * patches_w

        # initialize cls token and positional embeddings as trainable weights
        if self.cls_token is None:
            self.cls_token = self.add_weight(
                name="cls_token",
                shape=(1, 1, self.embedding_dim),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                trainable=True,
            )

        if self.position_embedding is None:
            self.position_embedding = self.add_weight(
                name="position_embedding",
                shape=(1, self.num_patches + 1, self.embedding_dim),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                trainable=True,
            )

        super().build(input_shape)

    def call(self, img, training=False, return_sequence=False):
        # ensure weights exist if build wasn't called explicitly
        if self.cls_token is None or self.position_embedding is None:
            self.build(img.shape)

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
        blank_bias=0.0,
    ):

        super().__init__()

        self.encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=embedding_dim,
            num_transformer_blocks=num_transformer_blocks,
        )
        self.token_projection = tf.keras.layers.Dense(vocab_size)
        self._vocab_size = vocab_size
        self._blank_bias = blank_bias

    def build(self, input_shape):
        # ensure encoder weights exist and compute token projection bias
        try:
            self.encoder.build(input_shape)
        except Exception:
            pass

        # build token projection by calling it once with a dummy shape matching patch tokens
        # input_shape: (batch, H, W, C) -> encoder returns (batch, patches+1, dim)
        batch = 1
        h = input_shape[1] if len(input_shape) > 1 and input_shape[1] is not None else self.encoder.image_size
        w = input_shape[2] if len(input_shape) > 2 and input_shape[2] is not None else self.encoder.image_size
        dummy = tf.zeros([batch, h, w, 3])
        try:
            seq = self.encoder(dummy, training=False, return_sequence=True)
            patch_tokens = seq[:, 1:, :]
            _ = self.token_projection(patch_tokens)
        except Exception:
            # final fallback: build projection directly
            self.token_projection.build((batch, max(1, self.encoder.num_patches), self.encoder.embedding_dim))

        # Keep the blank class neutral by default. CTC already rewards blank
        # alignments heavily, so a positive blank bias can make early training
        # collapse into empty-string predictions.
        if self._blank_bias != 0.0:
            try:
                bias = self.token_projection.bias
                if bias is not None:
                    b = tf.zeros_like(bias)
                    blank_index = tf.cast(self._vocab_size - 1, tf.int32)
                    b = tf.tensor_scatter_nd_add(
                        b, [[blank_index]], [tf.cast(self._blank_bias, b.dtype)]
                    )
                    self.token_projection.bias.assign(b)
            except Exception:
                pass

    def call(self, img, training=False):
        sequence = self.encoder(img, training=training, return_sequence=True)

        patch_tokens = sequence[:, 1:, :]
        logits = self.token_projection(patch_tokens)

        return logits


class OCRVocabulary:
    def __init__(self, characters=None):
        if characters is None:
            characters = (
                string.ascii_lowercase
                + string.ascii_uppercase
                + string.digits
                + " "
                + string.punctuation
            )

        self.characters = characters
        # Reserve 0 for padding. Map characters to ids starting at 1..N
        # For TensorFlow CTC, the blank symbol must be the last index (num_classes - 1).
        # We'll set blank_id = N+1.
        self.char_to_id = {char: index + 1 for index, char in enumerate(characters)}
        self.id_to_char = {index + 1: char for index, char in enumerate(characters)}
        self.blank_id = len(characters) + 1

    @property
    def size(self):
        return len(self.characters) + 2

    def encode(self, text):
        missing_chars = sorted({char for char in text if char not in self.char_to_id})
        if missing_chars:
            missing_display = ", ".join(repr(char) for char in missing_chars)
            raise ValueError(f"Unsupported OCR characters: {missing_display}")

        return [self.char_to_id[char] for char in text]

    def decode(self, token_ids):
        decoded_chars = []

        for token_id in token_ids:
            token_id = int(token_id)

            # skip padding (0) and blank symbol
            if token_id == 0 or token_id == self.blank_id:
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


def build_ctc_label_lengths_from_blank(labels, blank_id=None):
    if blank_id is None:
        return tf.math.count_nonzero(labels, axis=1, keepdims=True, dtype=tf.int32)
    # count tokens that are not equal to padding (0) and not equal to blank_id
    labels = tf.convert_to_tensor(labels)
    pad_mask = tf.not_equal(labels, tf.cast(0, labels.dtype))
    if blank_id is None:
        mask = pad_mask
    else:
        mask = tf.logical_and(pad_mask, tf.not_equal(labels, tf.cast(blank_id, labels.dtype)))

    return tf.reduce_sum(tf.cast(mask, tf.int32), axis=1, keepdims=True)


def ctc_loss(labels, logits, blank_id=None):
    batch_size = tf.shape(logits)[0]
    time_steps = tf.shape(logits)[1]

    input_lengths = tf.squeeze(build_ctc_input_lengths(batch_size, time_steps), axis=-1)
    label_lengths = tf.squeeze(
        build_ctc_label_lengths_from_blank(labels, blank_id=blank_id), axis=-1
    )
    if blank_id is None:
        blank_id = tf.shape(logits)[-1] - 1

    loss = tf.nn.ctc_loss(
        labels=tf.cast(labels, tf.int32),
        logits=logits,
        label_length=label_lengths,
        logit_length=input_lengths,
        logits_time_major=False,
        blank_index=tf.cast(blank_id, tf.int32),
    )

    return tf.expand_dims(loss, axis=-1)


def greedy_ctc_decode(logits, vocabulary):
    logits = tf.convert_to_tensor(logits)
    token_ids = tf.argmax(logits, axis=-1, output_type=tf.int32).numpy()

    decoded_texts = []
    for row in token_ids:
        collapsed = []
        previous_token = None

        for token_id in row:
            token_id = int(token_id)

            # skip blanks
            if token_id == vocabulary.blank_id:
                previous_token = token_id
                continue

            # collapse consecutive duplicates
            if previous_token == token_id:
                continue

            collapsed.append(token_id)
            previous_token = token_id

        decoded_text = vocabulary.decode(collapsed)
        decoded_texts.append(decoded_text)

    return decoded_texts


def beam_search_ctc_decode(logits, vocabulary, beam_width=10, top_paths=1):
    """Decode logits with TensorFlow CTC beam search decoder and return top path strings.

    logits: array-like (batch, time, num_classes)
    """
    logits = tf.convert_to_tensor(logits)
    # TF CTC expects time-major inputs: (time, batch, num_classes)
    inputs = tf.transpose(logits, perm=[1, 0, 2])
    batch_size = tf.shape(logits)[0]
    time_steps = tf.shape(logits)[1]

    seq_len = tf.fill([batch_size], tf.cast(time_steps, tf.int32))

    decoded, log_prob = tf.nn.ctc_beam_search_decoder(inputs, seq_len, beam_width=beam_width, top_paths=top_paths)

    # decoded is a list of SparseTensors (top_paths long). We'll take the first path.
    sparse_decoded = decoded[0]
    # use 0 as default padding value so downstream decode logic can skip 0 and blank_id
    dense = tf.sparse.to_dense(sparse_decoded, default_value=0)
    dense_np = dense.numpy()

    texts = []
    for row in dense_np:
        token_ids = [int(t) for t in row if int(t) != vocabulary.blank_id]
        texts.append(vocabulary.decode(token_ids))

    return texts
