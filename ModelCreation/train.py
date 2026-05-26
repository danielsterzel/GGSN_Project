"""Trening OCR opartego o ViT z CTC.

Skrypt obsługuje 2 tryby:
1. Demo (domyślnie, bez manifestu) - szybki test przepływu treningu.
2. Realny trening na danych z manifestu CSV.

Przykład uruchomienia realnego treningu:
python ModelCreation/train.py --manifest data/manifest.csv --root-dir data --epochs 10
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys

import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from DataManagment.OCRData import OCRDatasetConfig, OCRSample, build_dataset, read_manifest

try:
	from ModelCreation.ViT import OCRVocabulary, ViTOCR, ctc_loss, greedy_ctc_decode
except ImportError:
	# Fallback for running from within ModelCreation/ directory.
	from ViT import OCRVocabulary, ViTOCR, ctc_loss, greedy_ctc_decode


def build_demo_model(image_size: int = 256, patch_size: int = 16):
	vocabulary = OCRVocabulary()
	model = ViTOCR(
		vocab_size=vocabulary.size,
		image_size=image_size,
		patch_size=patch_size,
	)
	return model, vocabulary


def train_step(model, images, labels, optimizer):
	with tf.GradientTape() as tape:
		logits = model(images, training=True)
		loss_value = tf.reduce_mean(ctc_loss(labels, logits))

	gradients = tape.gradient(loss_value, model.trainable_variables)
	optimizer.apply_gradients(zip(gradients, model.trainable_variables))

	return loss_value, logits


def eval_step(model, images, labels):
	logits = model(images, training=False)
	loss_value = tf.reduce_mean(ctc_loss(labels, logits))
	return loss_value, logits


def _label_tensor_to_texts(labels, vocabulary):
	texts = []
	for row in labels.numpy():
		token_ids = [int(token_id) for token_id in row if int(token_id) != vocabulary.blank_id]
		texts.append(vocabulary.decode(token_ids))
	return texts


def _edit_distance(source: str, target: str) -> int:
	if source == target:
		return 0
	if not source:
		return len(target)
	if not target:
		return len(source)

	prev = list(range(len(target) + 1))
	for i, src_char in enumerate(source, start=1):
		curr = [i]
		for j, tgt_char in enumerate(target, start=1):
			insert_cost = curr[j - 1] + 1
			delete_cost = prev[j] + 1
			replace_cost = prev[j - 1] + (0 if src_char == tgt_char else 1)
			curr.append(min(insert_cost, delete_cost, replace_cost))
		prev = curr

	return prev[-1]


def _sequence_metrics(labels, logits, vocabulary):
	pred_texts = greedy_ctc_decode(logits, vocabulary)
	target_texts = _label_tensor_to_texts(labels, vocabulary)

	if not target_texts:
		return 0.0, 0.0

	seq_matches = 0
	char_acc_sum = 0.0

	for pred_text, target_text in zip(pred_texts, target_texts):
		if pred_text == target_text:
			seq_matches += 1

		dist = _edit_distance(pred_text, target_text)
		norm = max(1, len(target_text))
		char_acc_sum += max(0.0, 1.0 - (dist / norm))

	seq_acc = seq_matches / len(target_texts)
	char_acc = char_acc_sum / len(target_texts)
	return char_acc, seq_acc


def _split_samples(samples, val_split=0.1, seed=42):
	if not samples:
		return [], []

	rng = random.Random(seed)
	shuffled = samples[:]
	rng.shuffle(shuffled)

	val_count = int(len(shuffled) * val_split)
	if val_split > 0 and val_count == 0 and len(shuffled) > 1:
		val_count = 1
	if val_count >= len(shuffled):
		val_count = max(0, len(shuffled) - 1)

	val_samples = shuffled[:val_count]
	train_samples = shuffled[val_count:]
	return train_samples, val_samples


def _get_learning_rate(optimizer):
	lr_value = optimizer.learning_rate
	if hasattr(lr_value, "numpy"):
		return float(lr_value.numpy())
	return float(lr_value)


def _set_learning_rate(optimizer, value):
	if hasattr(optimizer.learning_rate, "assign"):
		optimizer.learning_rate.assign(value)
	else:
		optimizer.learning_rate = value


def _resolve_manifest_path(manifest_arg: str) -> Path:
	manifest_path = Path(manifest_arg)
	if manifest_path.is_file():
		return manifest_path

	manifest_from_repo = REPO_ROOT / manifest_path
	if manifest_from_repo.is_file():
		return manifest_from_repo

	available_csv = sorted(REPO_ROOT.glob("**/*.csv"))
	preview = "\n".join(f"- {p.relative_to(REPO_ROOT)}" for p in available_csv[:10])
	if not preview:
		preview = "- (brak plikow CSV w repo)"

	raise FileNotFoundError(
		"Nie znaleziono manifestu CSV.\n"
		f"Podana sciezka: {manifest_arg}\n"
		f"Sprawdzane lokalizacje:\n"
		f"- {manifest_path.resolve()}\n"
		f"- {(REPO_ROOT / manifest_path).resolve()}\n"
		"Dostepne pliki CSV w repo:\n"
		f"{preview}\n"
		"Utworz plik manifestu albo podaj poprawna sciezke przez --manifest."
	)


def run_demo(learning_rate=1e-4):
	model, vocabulary = build_demo_model()
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	dummy_samples = [
		OCRSample(image_path="dummy_1.png", text="test"),
		OCRSample(image_path="dummy_2.png", text="ocr"),
	]
	dummy_images = tf.random.normal((2, 256, 256, 3))
	dummy_labels = tf.ragged.constant(
		[vocabulary.encode(sample.text) for sample in dummy_samples], dtype=tf.int32
	).to_tensor(default_value=vocabulary.blank_id)

	loss_value, logits = train_step(
		model=model,
		images=dummy_images,
		labels=dummy_labels,
		optimizer=optimizer,
	)

	predictions = greedy_ctc_decode(logits, vocabulary)
	print("[DEMO] loss:", float(loss_value.numpy()))
	print("[DEMO] predictions:", predictions)


def train_on_manifest(args):
	tf.random.set_seed(args.seed)

	model, vocabulary = build_demo_model(
		image_size=args.image_size,
		patch_size=args.patch_size,
	)
	optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

	manifest_path = _resolve_manifest_path(args.manifest)
	print(f"Uzywam manifestu: {manifest_path}")

	samples = read_manifest(
		csv_path=manifest_path,
		root_dir=args.root_dir,
		image_column=args.image_column,
		text_column=args.text_column,
	)

	if not samples:
		raise ValueError("Manifest nie zawiera żadnych próbek.")

	if len(samples) < 50:
		print(
			"UWAGA: bardzo mały zbiór danych "
			f"({len(samples)} próbek). Oczekuj niestabilnych metryk i szybkiego plateau."
		)

	effective_val_split = args.val_split
	if len(samples) < 10 and args.val_split > 0:
		print(
			"UWAGA: dla <10 próbek automatycznie ustawiam val_split=0.0, "
			"aby nie tracić danych treningowych."
		)
		effective_val_split = 0.0

	train_samples, val_samples = _split_samples(
		samples,
		val_split=effective_val_split,
		seed=args.seed,
	)
	if not train_samples:
		raise ValueError("Brak próbek treningowych po podziale train/val.")

	data_config = OCRDatasetConfig(
		image_size=(args.image_size, args.image_size),
		batch_size=args.batch_size,
		shuffle_buffer_size=max(args.batch_size * 8, 64),
		cache=args.cache,
		augment=not args.disable_augment,
		augment_brightness_delta=args.augment_brightness_delta,
		augment_contrast_lower=args.augment_contrast_lower,
		augment_contrast_upper=args.augment_contrast_upper,
		augment_noise_stddev=args.augment_noise_stddev,
	)
	print(
		"Augmentacje: "
		f"enabled={data_config.augment} "
		f"brightness={data_config.augment_brightness_delta} "
		f"contrast=({data_config.augment_contrast_lower},{data_config.augment_contrast_upper}) "
		f"noise_stddev={data_config.augment_noise_stddev}"
	)

	train_ds = build_dataset(train_samples, vocabulary, data_config, training=True)
	val_ds = None
	if val_samples:
		val_ds = build_dataset(val_samples, vocabulary, data_config, training=False)

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
	checkpoint_manager = tf.train.CheckpointManager(
		checkpoint,
		str(output_dir / "checkpoints"),
		max_to_keep=3,
	)

	best_val_loss = float("inf")
	best_path = output_dir / "best.weights.h5"
	epochs_without_lr_improvement = 0

	print(f"Start treningu: train={len(train_samples)} val={len(val_samples)}")
	for epoch in range(1, args.epochs + 1):
		train_losses = []
		train_norm_losses = []
		train_char_accs = []
		train_seq_accs = []

		for images, labels in train_ds:
			loss_value, logits = train_step(model, images, labels, optimizer)
			train_losses.append(float(loss_value.numpy()))
			time_steps = float(tf.shape(logits)[1].numpy())
			train_norm_losses.append(float(loss_value.numpy()) / max(1.0, time_steps))
			char_acc, seq_acc = _sequence_metrics(labels, logits, vocabulary)
			train_char_accs.append(char_acc)
			train_seq_accs.append(seq_acc)

		train_loss = sum(train_losses) / max(1, len(train_losses))
		train_loss_norm = sum(train_norm_losses) / max(1, len(train_norm_losses))
		train_char_acc = sum(train_char_accs) / max(1, len(train_char_accs))
		train_seq_acc = sum(train_seq_accs) / max(1, len(train_seq_accs))

		if val_ds is not None:
			val_losses = []
			val_norm_losses = []
			val_char_accs = []
			val_seq_accs = []
			for images, labels in val_ds:
				val_loss_value, val_logits = eval_step(model, images, labels)
				val_losses.append(float(val_loss_value.numpy()))
				time_steps = float(tf.shape(val_logits)[1].numpy())
				val_norm_losses.append(float(val_loss_value.numpy()) / max(1.0, time_steps))
				char_acc, seq_acc = _sequence_metrics(labels, val_logits, vocabulary)
				val_char_accs.append(char_acc)
				val_seq_accs.append(seq_acc)

			val_loss = sum(val_losses) / max(1, len(val_losses))
			val_loss_norm = sum(val_norm_losses) / max(1, len(val_norm_losses))
			val_char_acc = sum(val_char_accs) / max(1, len(val_char_accs))
			val_seq_acc = sum(val_seq_accs) / max(1, len(val_seq_accs))
		else:
			val_loss = train_loss
			val_loss_norm = train_loss_norm
			val_char_acc = train_char_acc
			val_seq_acc = train_seq_acc

		print(
			f"Epoch {epoch:03d}/{args.epochs} "
			f"train_loss={train_loss:.4f} train_loss_norm={train_loss_norm:.4f} "
			f"train_char_acc={train_char_acc:.4f} train_seq_acc={train_seq_acc:.4f} "
			f"val_loss={val_loss:.4f} val_loss_norm={val_loss_norm:.4f} "
			f"val_char_acc={val_char_acc:.4f} val_seq_acc={val_seq_acc:.4f}"
		)

		checkpoint_manager.save()

		if best_val_loss - val_loss > args.min_delta:
			best_val_loss = val_loss
			epochs_without_lr_improvement = 0
			model.save_weights(best_path)
			print(f"Zapisano najlepsze wagi: {best_path}")
		else:
			epochs_without_lr_improvement += 1

			if args.lr_scheduler == "plateau" and epochs_without_lr_improvement >= args.lr_patience:
				current_lr = _get_learning_rate(optimizer)
				new_lr = max(current_lr * args.lr_factor, args.min_lr)
				if new_lr < current_lr:
					_set_learning_rate(optimizer, new_lr)
					print(
						f"LR scheduler: brak poprawy przez {args.lr_patience} epok, "
						f"zmniejszam lr {current_lr:.2e} -> {new_lr:.2e}"
					)
				epochs_without_lr_improvement = 0

	# Krótki podgląd dekodowania na jednej batchy treningowej.
	for images, labels in train_ds.take(1):
		logits = model(images, training=False)
		decoded = greedy_ctc_decode(logits, vocabulary)
		print("Przykładowe predykcje:", decoded[: min(3, len(decoded))])

	final_path = output_dir / "final.weights.h5"
	model.save_weights(final_path)
	print(f"Zapisano finalne wagi: {final_path}")


def build_arg_parser():
	parser = argparse.ArgumentParser(description="Trening ViT OCR z CTC")
	parser.add_argument("--manifest", type=str, default=None, help="Ścieżka do CSV manifestu")
	parser.add_argument("--root-dir", type=str, default=None, help="Root dla ścieżek obrazów")
	parser.add_argument("--image-column", type=str, default="image_path")
	parser.add_argument("--text-column", type=str, default="text")
	parser.add_argument("--output-dir", type=str, default="artifacts")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--learning-rate", type=float, default=1e-4)
	parser.add_argument("--image-size", type=int, default=256)
	parser.add_argument("--patch-size", type=int, default=16)
	parser.add_argument("--val-split", type=float, default=0.1)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--cache", action="store_true", help="Cache datasetu tf.data")
	parser.add_argument(
		"--disable-augment",
		action="store_true",
		help="Wyłącza augmentacje obrazów podczas treningu.",
	)
	parser.add_argument(
		"--augment-brightness-delta",
		type=float,
		default=0.08,
		help="Maksymalna zmiana jasności dla augmentacji.",
	)
	parser.add_argument(
		"--augment-contrast-lower",
		type=float,
		default=0.9,
		help="Dolna granica kontrastu dla augmentacji.",
	)
	parser.add_argument(
		"--augment-contrast-upper",
		type=float,
		default=1.1,
		help="Górna granica kontrastu dla augmentacji.",
	)
	parser.add_argument(
		"--augment-noise-stddev",
		type=float,
		default=0.02,
		help="Odchylenie standardowe szumu dodawanego do obrazu.",
	)
	parser.add_argument(
		"--min-delta",
		type=float,
		default=1e-4,
		help="Minimalna poprawa val_loss uznawana za postęp.",
	)
	parser.add_argument(
		"--lr-scheduler",
		type=str,
		choices=["none", "plateau"],
		default="plateau",
		help="Strategia schedulera learning rate.",
	)
	parser.add_argument(
		"--lr-patience",
		type=int,
		default=2,
		help="Liczba epok bez poprawy przed redukcją LR (dla plateau).",
	)
	parser.add_argument(
		"--lr-factor",
		type=float,
		default=0.5,
		help="Mnożnik redukcji LR dla plateau.",
	)
	parser.add_argument(
		"--min-lr",
		type=float,
		default=1e-6,
		help="Minimalna wartość learning rate.",
	)
	return parser


def main(cli_args=None):
	parser = build_arg_parser()
	args = parser.parse_args(cli_args)

	if args.manifest:
		train_on_manifest(args)
	else:
		print("Brak --manifest, uruchamiam tryb demo.")
		run_demo(learning_rate=args.learning_rate)


if __name__ == "__main__":
	main()
