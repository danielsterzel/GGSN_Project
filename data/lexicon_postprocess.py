import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DataManagment.OCRData import OCRDatasetConfig, read_manifest, build_dataset
from ModelCreation.ViT import OCRVocabulary, ViTOCR, greedy_ctc_decode, beam_search_ctc_decode


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(insert, delete, replace))
        prev = curr
    return prev[-1]


def normalize(a: str, b: str) -> float:
    # normalized edit distance (0..1)
    maxlen = max(1, len(b))
    return edit_distance(a, b) / maxlen


def main():
    artifacts = REPO_ROOT / "artifacts_run_after_changes"
    cfg_path = artifacts / "model_config.json"
    weights = artifacts / "final.weights.h5"
    if not cfg_path.exists() or not weights.exists():
        print("Missing artifacts in", artifacts)
        return

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    vocab = OCRVocabulary(characters=cfg.get("vocabulary"))

    model = ViTOCR(
        vocab_size=vocab.size,
        image_size=cfg.get("image_size", 256),
        patch_size=cfg.get("patch_size", 16),
        embedding_dim=cfg.get("embedding_dim", 256),
        num_transformer_blocks=cfg.get("num_transformer_blocks", 4),
    )
    model.build((None, cfg.get("image_size", 256), cfg.get("image_size", 256), 3))
    model.load_weights(str(weights))

    samples = read_manifest(REPO_ROOT / "data" / "manifest.csv", root_dir=REPO_ROOT / "data")
    unique_texts = sorted({s.text for s in samples})

    data_config = OCRDatasetConfig(image_size=(cfg.get("image_size", 256), cfg.get("image_size", 256)), batch_size=8)
    ds = build_dataset(samples[:64], vocab, data_config, training=False)

    for images, labels in ds.take(1):
        logits = model(images, training=False)
        greedy = greedy_ctc_decode(logits, vocab)
        beam = beam_search_ctc_decode(logits, vocab, beam_width=8)

        corrected = []
        for text in greedy:
            best = text
            best_score = 1.0
            for cand in unique_texts:
                score = normalize(text, cand)
                if score < best_score:
                    best_score = score
                    best = cand
            # accept correction only if sufficiently close
            if best_score < 0.4 and best != text:
                corrected.append((text, best, best_score))
            else:
                corrected.append((text, None, best_score))

        print("Sample comparisons (greedy -> corrected if any):")
        for i, (g, corr, score) in enumerate(corrected[:10]):
            print(i, "greedy=", repr(g), "->", repr(corr) if corr else "(no change)", "score=", f"{score:.2f}")

        print("Beam samples:")
        for i, b in enumerate(beam[:10]):
            print(i, b)
        break


if __name__ == '__main__':
    main()
