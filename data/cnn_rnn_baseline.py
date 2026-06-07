from __future__ import annotations

import json
import sys
from pathlib import Path

import tensorflow as tf
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DataManagment.OCRData import OCRDatasetConfig, read_manifest, build_dataset
from ModelCreation.ViT import OCRVocabulary, ctc_loss, greedy_ctc_decode


def build_crnn(vocab_size: int, image_size=(128, 128)):
    H, W = image_size
    inp = tf.keras.layers.Input(shape=(H, W, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inp)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D((2, 2))(x)
    # collapse height -> time dimension along width
    x = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=1))(x)  # (B, W//4, C)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    logits = tf.keras.layers.Dense(vocab_size, activation=None)(x)  # (B, T, vocab)
    return tf.keras.Model(inputs=inp, outputs=logits)


def train_baseline(manifest_path: str, root_dir: str, epochs=10, image_size=(128, 128), batch_size=8):
    samples = read_manifest(manifest_path, root_dir=Path(root_dir))
    all_text = "".join(s.text for s in samples)
    unique_chars = "".join(sorted(set(all_text)))
    vocab = OCRVocabulary(characters=unique_chars)

    # estimate time-steps produced by the CRNN: we downsample width by factor 4 (two MaxPool2D)
    time_steps = max(1, image_size[1] // 4)
    max_label_len = max(1, time_steps // 2)
    filtered = [s for s in samples if len(s.text) <= max_label_len]
    print(f"Original samples={len(samples)} filtered_by_label_len<={max_label_len} => {len(filtered)}")

    data_config = OCRDatasetConfig(image_size=image_size, batch_size=batch_size)
    ds = build_dataset(filtered, vocab, data_config, training=True)

    model = build_crnn(vocab_size=vocab.size, image_size=image_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = tf.reduce_mean(ctc_loss(labels, logits, blank_id=vocab.blank_id))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, logits

    for epoch in range(1, epochs + 1):
        losses = []
        for images, labels in ds:
            loss, logits = train_step(images, labels)
            losses.append(float(loss.numpy()))
        print(f"Epoch {epoch}/{epochs} loss={sum(losses)/len(losses):.4f}")

    # quick decode on one batch
    for images, labels in ds.take(1):
        logits = model(images, training=False)
        preds = greedy_ctc_decode(logits, vocab)
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        argmax_ids = np.argmax(probs, axis=-1)
        max_probs = np.max(probs, axis=-1)
        targets = []
        for row in labels.numpy():
            token_ids = [int(t) for t in row if int(t) != vocab.blank_id]
            targets.append(vocab.decode(token_ids))
        for i in range(min(8, len(preds))):
            mean_max = float(np.mean(max_probs[i]))
            blank_prop = float(np.mean((argmax_ids[i] == vocab.blank_id).astype(np.float32)))
            print(f"SAMPLE {i} target={targets[i]} pred={preds[i]} mean_max_prob={mean_max:.4f} blank_prop={blank_prop:.3f}")


if __name__ == "__main__":
    manifest = str(REPO_ROOT / "data" / "manifest.csv")
    root = str(REPO_ROOT / "data")
    train_baseline(manifest, root, epochs=10, image_size=(128, 128), batch_size=8)
