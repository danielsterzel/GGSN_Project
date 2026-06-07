import json
from pathlib import Path
import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DataManagment.OCRData import OCRDatasetConfig, read_manifest, build_dataset
from ModelCreation.ViT import OCRVocabulary, ViTOCR, greedy_ctc_decode, beam_search_ctc_decode


def load_config(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_batch(model, vocab, images, labels, top_paths=3):
    logits = model(images, training=False)
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    argmax_ids = np.argmax(probs, axis=-1)
    max_probs = np.max(probs, axis=-1)

    greedy = greedy_ctc_decode(logits, vocab)
    beam = beam_search_ctc_decode(logits, vocab, beam_width=8)

    target_texts = []
    for row in labels.numpy():
        token_ids = [int(token_id) for token_id in row if int(token_id) != vocab.blank_id]
        target_texts.append(vocab.decode(token_ids))

    for i in range(min(8, images.shape[0])):
        ids = argmax_ids[i]
        mean_max = float(np.mean(max_probs[i]))
        blank_prop = float(np.mean((ids == vocab.blank_id).astype(np.float32)))
        print(f"SAMPLE {i}")
        print("  target:", target_texts[i])
        print("  greedy:", greedy[i])
        print("  beam:", beam[i])
        print("  argmax_ids[:40]:", ids[:40].tolist())
        print(f"  mean_max_prob={mean_max:.4f} blank_prop={blank_prop:.3f}")
        print()


def main():
    import sys

    # allow passing artifacts dir as first arg, else fallback to artifacts_diag
    artifacts = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "artifacts_diag"
    cfg_path = artifacts / "model_config.json"
    weights = artifacts / "best.weights.h5"

    # Fallback: if model_config.json not present yet, build vocabulary from manifest
    if cfg_path.exists():
        cfg = load_config(cfg_path)
        vocab = OCRVocabulary(characters=cfg.get("vocabulary"))
    else:
        print("model_config.json not found; building vocabulary from manifest as fallback")
        samples = read_manifest(REPO_ROOT / "data" / "manifest.csv", root_dir=REPO_ROOT / "data")
        all_text = "".join(s.text for s in samples)
        unique_chars = "".join(sorted(set(all_text)))
        cfg = {"image_size": 256, "patch_size": 16, "embedding_dim": 256, "num_transformer_blocks": 4}
        vocab = OCRVocabulary(characters=unique_chars)

    model = ViTOCR(
        vocab_size=vocab.size,
        image_size=cfg.get("image_size", 256),
        patch_size=cfg.get("patch_size", 16),
        embedding_dim=cfg.get("embedding_dim", 256),
        num_transformer_blocks=cfg.get("num_transformer_blocks", 4),
    )

    model.build((None, cfg.get("image_size", 256), cfg.get("image_size", 256), 3))
    if weights.exists():
        model.load_weights(str(weights))
        print("Loaded weights:", weights)
    else:
        print("Weights not found at", weights, "— continuing with randomly initialized model for inspection")

    samples = read_manifest(REPO_ROOT / "data" / "manifest.csv", root_dir=REPO_ROOT / "data")
    data_config = OCRDatasetConfig(image_size=(cfg.get("image_size", 256), cfg.get("image_size", 256)), batch_size=8)
    ds = build_dataset(samples[:64], vocab, data_config, training=False)

    for images, labels in ds.take(1):
        analyze_batch(model, vocab, images, labels)
        break


if __name__ == '__main__':
    main()
