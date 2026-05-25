from dataclasses import dataclass
from pathlib import Path
import csv

import tensorflow as tf


@dataclass(frozen=True)
class OCRSample:
	image_path: str
	text: str


@dataclass(frozen=True)
class OCRDatasetConfig:
	image_size: tuple[int, int] = (256, 256)
	batch_size: int = 8
	shuffle_buffer_size: int = 256
	cache: bool = False


def read_manifest(
	csv_path,
	root_dir=None,
	image_column="image_path",
	text_column="text",
	encoding="utf-8-sig",
):
	csv_path = Path(csv_path)
	root_dir = Path(root_dir) if root_dir is not None else csv_path.parent

	samples = []

	with csv_path.open("r", newline="", encoding=encoding) as file_handle:
		reader = csv.DictReader(file_handle)

		for row in reader:
			image_path = Path(row[image_column])
			if not image_path.is_absolute():
				image_path = root_dir / image_path

			samples.append(
				OCRSample(
					image_path=str(image_path),
					text=row[text_column],
				)
			)

	return samples


def load_image(image_path, image_size=(256, 256)):
	image_bytes = tf.io.read_file(image_path)
	image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
	image.set_shape([None, None, 3])
	image = tf.image.resize(image, image_size)
	image = tf.cast(image, tf.float32) / 255.0

	return image


def encode_text(text, vocabulary):
	encoded = tf.py_function(
		func=lambda value: tf.convert_to_tensor(
			vocabulary.encode(value.numpy().decode("utf-8")), dtype=tf.int32
		),
		inp=[text],
		Tout=tf.int32,
	)
	encoded.set_shape([None])

	return encoded


def build_dataset(samples, vocabulary, config: OCRDatasetConfig, training=True):
	image_paths = [sample.image_path for sample in samples]
	texts = [sample.text for sample in samples]

	dataset = tf.data.Dataset.from_tensor_slices((image_paths, texts))

	if training:
		dataset = dataset.shuffle(
			min(len(samples), config.shuffle_buffer_size), reshuffle_each_iteration=True
		)

	def _load_example(image_path, text):
		image = load_image(image_path, config.image_size)
		label = encode_text(text, vocabulary)

		return image, label

	dataset = dataset.map(_load_example, num_parallel_calls=tf.data.AUTOTUNE)

	if config.cache:
		dataset = dataset.cache()

	dataset = dataset.padded_batch(
		config.batch_size,
		padded_shapes=([config.image_size[0], config.image_size[1], 3], [None]),
		padding_values=(tf.constant(0.0, tf.float32), tf.constant(0, tf.int32)),
		drop_remainder=False,
	)

	return dataset.prefetch(tf.data.AUTOTUNE)
