"""Minimalny punkt startowy do treningu OCR opartego o ViT jako encoder.

Ten plik nie trenuje jeszcze pełnego OCR end-to-end na realnym zbiorze,
ale pokazuje potrzebny szkielet: model, słownik znaków i CTC loss.
"""

import tensorflow as tf

from DataManagment.OCRData import OCRDatasetConfig, OCRSample, build_dataset
from ViT import OCRVocabulary, ViTOCR, ctc_loss, greedy_ctc_decode


def build_demo_model():
	vocabulary = OCRVocabulary()
	model = ViTOCR(vocab_size=vocabulary.size)

	return model, vocabulary


def train_step(model, images, labels, optimizer):
	with tf.GradientTape() as tape:
		logits = model(images, training=True)
		loss_value = tf.reduce_mean(ctc_loss(labels, logits))

	gradients = tape.gradient(loss_value, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	return loss_value, logits


def main():
	model, vocabulary = build_demo_model()
	optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

	dummy_samples = [
		OCRSample(image_path="dummy_1.png", text="test"),
		OCRSample(image_path="dummy_2.png", text="ocr"),
	]
	dummy_images = tf.random.normal((2, 256, 256, 3))
	dummy_labels = tf.ragged.constant(
		[vocabulary.encode(sample.text) for sample in dummy_samples], dtype=tf.int32
	).to_tensor(default_value=vocabulary.blank_id)

	loss_value, logits = train_step(
		model,
		dummy_images,
		dummy_labels,
		optimizer,
	)

	predictions = greedy_ctc_decode(logits, vocabulary)

	print("loss:", float(loss_value.numpy()))
	print("predictions:", predictions)

	# Przykład użycia loadera na realnym manifest.csv:
	# samples = [OCRSample(image_path="data/img_1.png", text="hello")]
	# dataset = build_dataset(samples, vocabulary, OCRDatasetConfig())
	# for batch_images, batch_labels in dataset.take(1):
	#     loss_value, logits = train_step(model, batch_images, batch_labels, optimizer)


if __name__ == "__main__":
	main()
