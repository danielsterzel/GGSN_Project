"""Evaluate OCR weights against a manifest CSV.

Example:
python ModelCreation/evaluate_ocr.py --weights best.weights.h5 final.weights.h5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(REPO_ROOT))

from DataManagment.OCRData import OCRDatasetConfig, build_dataset, read_manifest
from ModelCreation.ViT import OCRVocabulary, ViTOCR, greedy_ctc_decode


def _edit_distance(source: str, target: str) -> int:
	if source == target:
		return 0
	if not source:
		return len(target)
	if not target:
		return len(source)

	previous = list(range(len(target) + 1))
	for i, source_char in enumerate(source, start=1):
		current = [i]
		for j, target_char in enumerate(target, start=1):
			current.append(
				min(
					current[j - 1] + 1,
					previous[j] + 1,
					previous[j - 1] + (0 if source_char == target_char else 1),
				)
			)
		previous = current

	return previous[-1]


def _word_distance(source: str, target: str) -> int:
	source_words = source.split()
	target_words = target.split()
	if source_words == target_words:
		return 0

	previous = list(range(len(target_words) + 1))
	for i, source_word in enumerate(source_words, start=1):
		current = [i]
		for j, target_word in enumerate(target_words, start=1):
			current.append(
				min(
					current[j - 1] + 1,
					previous[j] + 1,
					previous[j - 1] + (0 if source_word == target_word else 1),
				)
			)
		previous = current

	return previous[-1]


def _split_samples(samples, val_split: float, seed: int):
	shuffled = samples[:]
	random.Random(seed).shuffle(shuffled)

	val_count = int(len(shuffled) * val_split)
	if val_split > 0 and val_count == 0 and len(shuffled) > 1:
		val_count = 1
	if val_count >= len(shuffled):
		val_count = max(0, len(shuffled) - 1)

	return shuffled[val_count:], shuffled[:val_count]


def _decode_labels(labels, vocabulary: OCRVocabulary) -> list[str]:
	texts = []
	for row in labels.numpy():
		token_ids = [
			int(token_id)
			for token_id in row
			if int(token_id) != 0 and int(token_id) != vocabulary.blank_id
		]
		texts.append(vocabulary.decode(token_ids))
	return texts


def _load_config(artifacts_dir: Path, manifest_path: Path, root_dir: Path | None):
	config_path = artifacts_dir / "model_config.json"
	if config_path.exists():
		return json.loads(config_path.read_text(encoding="utf-8"))

	samples = read_manifest(manifest_path, root_dir=root_dir)
	characters = "".join(sorted(set("".join(sample.text for sample in samples))))
	return {
		"image_size": 256,
		"patch_size": 16,
		"embedding_dim": 256,
		"num_transformer_blocks": 4,
		"vocabulary": characters or None,
	}


def _resolve_weights(weights_arg: str, artifacts_dir: Path) -> Path:
	weights_path = Path(weights_arg)
	if weights_path.is_file():
		return weights_path

	if not weights_path.is_absolute():
		for candidate in (artifacts_dir / weights_path, REPO_ROOT / weights_path):
			if candidate.is_file():
				return candidate

	return weights_path


def _build_model(config: dict, vocabulary: OCRVocabulary) -> ViTOCR:
	image_size = int(config.get("image_size", 256))
	model = ViTOCR(
		vocab_size=vocabulary.size,
		image_size=image_size,
		patch_size=int(config.get("patch_size", 16)),
		embedding_dim=int(config.get("embedding_dim", 256)),
		num_transformer_blocks=int(config.get("num_transformer_blocks", 4)),
		blank_bias=float(config.get("blank_bias", 0.0)),
	)
	model.build((None, image_size, image_size, 3))
	_ = model(tf.zeros((1, image_size, image_size, 3), dtype=tf.float32), training=False)
	return model


def _select_samples(samples, split: str, val_split: float, seed: int):
	train_samples, val_samples = _split_samples(samples, val_split=val_split, seed=seed)
	if split == "train":
		return train_samples
	if split == "validation":
		return val_samples
	return samples


def evaluate(model, vocabulary, samples, image_size: int, batch_size: int, examples: int):
	dataset = build_dataset(
		samples,
		vocabulary,
		OCRDatasetConfig(image_size=(image_size, image_size), batch_size=batch_size),
		training=False,
	)

	total = 0
	exact = 0
	empty = 0
	char_edits = 0
	char_total = 0
	word_edits = 0
	word_total = 0
	blank_argmax = 0
	argmax_total = 0
	max_prob_sum = 0.0
	prediction_length_sum = 0
	target_length_sum = 0
	sample_examples = []

	for images, labels in dataset:
		logits = model(images, training=False)
		predictions = greedy_ctc_decode(logits, vocabulary)
		targets = _decode_labels(labels, vocabulary)

		probs = tf.nn.softmax(logits, axis=-1).numpy()
		argmax_ids = np.argmax(probs, axis=-1)
		blank_argmax += int(np.sum(argmax_ids == vocabulary.blank_id))
		argmax_total += int(argmax_ids.size)
		max_prob_sum += float(np.sum(np.max(probs, axis=-1)))

		for prediction, target in zip(predictions, targets):
			total += 1
			exact += int(prediction == target)
			empty += int(prediction == "")
			char_edits += _edit_distance(prediction, target)
			char_total += len(target)
			word_edits += _word_distance(prediction, target)
			word_total += max(1, len(target.split()))
			prediction_length_sum += len(prediction)
			target_length_sum += len(target)

			if len(sample_examples) < examples:
				sample_examples.append((target, prediction))

	return {
		"samples": total,
		"exact_match": exact / max(1, total),
		"char_error_rate": char_edits / max(1, char_total),
		"char_accuracy": 1.0 - (char_edits / max(1, char_total)),
		"word_error_rate": word_edits / max(1, word_total),
		"empty_prediction_rate": empty / max(1, total),
		"blank_argmax_rate": blank_argmax / max(1, argmax_total),
		"mean_max_probability": max_prob_sum / max(1, argmax_total),
		"avg_target_length": target_length_sum / max(1, total),
		"avg_prediction_length": prediction_length_sum / max(1, total),
		"examples": sample_examples,
	}


def print_result(weights_path: Path, split: str, result: dict):
	print(f"\nWeights: {weights_path}")
	print(f"Split: {split}")
	print(f"Samples: {result['samples']}")
	print(f"Exact match: {result['exact_match']:.4f}")
	print(f"Character accuracy: {result['char_accuracy']:.4f}")
	print(f"Character error rate: {result['char_error_rate']:.4f}")
	print(f"Word error rate: {result['word_error_rate']:.4f}")
	print(f"Empty prediction rate: {result['empty_prediction_rate']:.4f}")
	print(f"Blank argmax rate: {result['blank_argmax_rate']:.4f}")
	print(f"Mean max probability: {result['mean_max_probability']:.4f}")
	print(f"Average target length: {result['avg_target_length']:.2f}")
	print(f"Average prediction length: {result['avg_prediction_length']:.2f}")

	if result["examples"]:
		print("Examples:")
		for target, prediction in result["examples"]:
			print(f"  target={target!r} prediction={prediction!r}")


def build_arg_parser():
	parser = argparse.ArgumentParser(description="Evaluate OCR weights.")
	parser.add_argument("--manifest", type=str, default="data/manifest.csv")
	parser.add_argument("--root-dir", type=str, default="data")
	parser.add_argument("--artifacts-dir", type=str, default="artifacts")
	parser.add_argument(
		"--weights",
		nargs="+",
		default=["best.weights.h5"],
		help="Weights files or names inside --artifacts-dir.",
	)
	parser.add_argument("--split", choices=["all", "train", "validation"], default="validation")
	parser.add_argument("--val-split", type=float, default=0.1)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--batch-size", type=int, default=16)
	parser.add_argument("--limit", type=int, default=None)
	parser.add_argument("--examples", type=int, default=8)
	return parser


def main(cli_args=None):
	args = build_arg_parser().parse_args(cli_args)

	manifest_path = Path(args.manifest)
	if not manifest_path.is_absolute():
		manifest_path = REPO_ROOT / manifest_path

	root_dir = Path(args.root_dir) if args.root_dir else None
	if root_dir is not None and not root_dir.is_absolute():
		root_dir = REPO_ROOT / root_dir

	artifacts_dir = Path(args.artifacts_dir)
	if not artifacts_dir.is_absolute():
		artifacts_dir = REPO_ROOT / artifacts_dir

	config = _load_config(artifacts_dir, manifest_path, root_dir)
	vocabulary = OCRVocabulary(characters=config.get("vocabulary") or None)
	image_size = int(config.get("image_size", 256))
	patch_size = int(config.get("patch_size", 16))
	time_steps = max(1, image_size // patch_size) ** 2

	samples = read_manifest(manifest_path, root_dir=root_dir)
	samples = [sample for sample in samples if len(sample.text) <= time_steps]
	samples = _select_samples(
		samples,
		split=args.split,
		val_split=args.val_split,
		seed=args.seed,
	)
	if args.limit is not None:
		samples = samples[: args.limit]

	print(
		"Dataset: "
		f"samples={len(samples)} split={args.split} "
		f"time_steps={time_steps} vocab_size={vocabulary.size}"
	)

	for weights_arg in args.weights:
		weights_path = _resolve_weights(weights_arg, artifacts_dir)
		if not weights_path.is_file():
			print(f"\nMissing weights: {weights_path}")
			continue

		model = _build_model(config, vocabulary)
		model.load_weights(str(weights_path))
		result = evaluate(
			model,
			vocabulary,
			samples,
			image_size=image_size,
			batch_size=args.batch_size,
			examples=args.examples,
		)
		print_result(weights_path, args.split, result)


if __name__ == "__main__":
	main()
