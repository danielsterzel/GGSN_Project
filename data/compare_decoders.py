import json
from pathlib import Path
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DataManagment.OCRData import OCRDatasetConfig, read_manifest, build_dataset
from ModelCreation.ViT import OCRVocabulary, ViTOCR, greedy_ctc_decode, beam_search_ctc_decode


def load_config(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    artifacts = REPO_ROOT / "artifacts_run_after_changes"
    config_path = artifacts / "model_config.json"
    if not config_path.exists():
        print("Model config not found:", config_path)
        return

    cfg = load_config(config_path)
    vocab_chars = cfg.get("vocabulary")
    vocab = OCRVocabulary(characters=vocab_chars)

    model = ViTOCR(
        vocab_size=vocab.size,
        image_size=cfg.get("image_size", 256),
        patch_size=cfg.get("patch_size", 16),
        embedding_dim=cfg.get("embedding_dim", 256),
        num_transformer_blocks=cfg.get("num_transformer_blocks", 4),
    )

    weights = artifacts / "final.weights.h5"
    if weights.exists():
        # build model before loading weights
        model.build((None, cfg.get("image_size", 256), cfg.get("image_size", 256), 3))
        model.load_weights(str(weights))
        print("Loaded weights from", weights)
    else:
        print("Weights not found:", weights)

    samples = read_manifest(REPO_ROOT / "data" / "manifest.csv", root_dir=REPO_ROOT / "data")
    data_config = OCRDatasetConfig(image_size=(cfg.get("image_size", 256), cfg.get("image_size", 256)), batch_size=8)
    ds = build_dataset(samples[:32], vocab, data_config, training=False)

    for images, labels in ds.take(1):
        logits = model(images, training=False)
        greedy = greedy_ctc_decode(logits, vocab)
        beam = beam_search_ctc_decode(logits, vocab, beam_width=8)
        print("Greedy:")
        print(greedy[:8])
        print("Beam:")
        print(beam[:8])
        break


if __name__ == '__main__':
    main()
