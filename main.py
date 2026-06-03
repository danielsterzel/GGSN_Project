from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
import json
import sys
from pathlib import Path
from typing import Optional
import io

import numpy as np
from PIL import Image
import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ModelCreation.ViT import OCRVocabulary, greedy_ctc_decode, ViTOCR
except Exception:
    from ViT import OCRVocabulary, greedy_ctc_decode, ViTOCR


MODEL_CANDIDATES = [
    REPO_ROOT / "artifacts" / "saved_model",
    REPO_ROOT / "ModelCreation" / "saved_model",
    REPO_ROOT / "artifacts" / "demo_saved_model",
]
CONFIG_CANDIDATES = [
    REPO_ROOT / "artifacts" / "model_config.json",
    REPO_ROOT / "ModelCreation" / "model_config.json",
]


def _load_model_config() -> dict:
    for config_path in CONFIG_CANDIDATES:
        if not config_path.exists():
            continue

        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Failed to load model config from {config_path}: {exc}")

    return {
        "image_size": 256,
        "patch_size": 16,
        "embedding_dim": 256,
        "num_transformer_blocks": 4,
        "vocabulary": None,
    }


def _find_model_path() -> Optional[Path]:
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    return None


MODEL_PATH = _find_model_path()
MODEL = None
MODEL_SIGNATURE = None
VOCAB = None
MODEL_CONFIG = _load_model_config()

if MODEL_PATH is not None:
    try:
        # Try Keras loader first (works for .keras or legacy formats supported by Keras)
        MODEL = tf.keras.models.load_model(str(MODEL_PATH))
        VOCAB = OCRVocabulary(characters=MODEL_CONFIG.get("vocabulary") or None)
        print(f"Loaded Keras model from {MODEL_PATH}")
    except Exception as e:
        print("Keras load_model failed, trying tf.saved_model.load:", e)
        try:
            loaded = tf.saved_model.load(str(MODEL_PATH))
            # serving_default is the typical inference signature
            if hasattr(loaded, 'signatures') and 'serving_default' in loaded.signatures:
                MODEL_SIGNATURE = loaded.signatures['serving_default']
            else:
                # fallback: try to use loaded as a callable
                MODEL_SIGNATURE = loaded

            VOCAB = OCRVocabulary(characters=MODEL_CONFIG.get("vocabulary") or None)
            print(f"Loaded SavedModel signatures from {MODEL_PATH}")
        except Exception as e2:
            print("Failed to load SavedModel via tf.saved_model.load:", e2)
        
    # Regardless whether a SavedModel signature exists, try to load HDF5 weights into ViTOCR
    if MODEL is None:
        image_size = int(MODEL_CONFIG.get("image_size", 256))
        patch_size = int(MODEL_CONFIG.get("patch_size", 16))
        embedding_dim = int(MODEL_CONFIG.get("embedding_dim", 256))
        num_transformer_blocks = int(MODEL_CONFIG.get("num_transformer_blocks", 4))
        WEIGHTS_CANDIDATES = [
            REPO_ROOT / "artifacts" / "best.weights.h5",
            REPO_ROOT / "artifacts" / "final.weights.h5",
            REPO_ROOT / "ModelCreation" / "final.weights.h5",
            REPO_ROOT / "ModelCreation" / "best.weights.h5",
            REPO_ROOT / "artifacts" / "demo.weights.h5",
        ]
        for w in WEIGHTS_CANDIDATES:
            if w.exists():
                try:
                    VOCAB = OCRVocabulary(characters=MODEL_CONFIG.get("vocabulary") or None)
                    model = ViTOCR(
                        vocab_size=VOCAB.size,
                        image_size=image_size,
                        patch_size=patch_size,
                        embedding_dim=embedding_dim,
                        num_transformer_blocks=num_transformer_blocks,
                    )
                    model.build((None, image_size, image_size, 3))
                    dummy = np.zeros((1, image_size, image_size, 3), dtype=np.float32)
                    _ = model(dummy, training=False)
                    model.load_weights(str(w))
                    MODEL = model
                    print(f"Loaded model weights from {w}")
                    break
                except Exception as e3:
                    print(f"Failed to load weights from {w}:", e3)
else:
    print("No SavedModel found at expected locations; /upload will run a random-model fallback if needed.")


def _preprocess_pil(img: Image.Image, image_size=(256, 256)) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(image_size)
    arr = np.asarray(img).astype("float32") / 255.0
    return arr


def predict_from_array(arr: np.ndarray):
    """Accepts HxWx3 numpy array, returns (decoded_texts, logits_shape)."""
    global MODEL, MODEL_SIGNATURE, VOCAB
    if MODEL is None and MODEL_SIGNATURE is None:
        raise RuntimeError("No model loaded. Train and save a model first.")

    batch = np.expand_dims(arr, 0)

    if MODEL is not None:
        logits = MODEL(batch, training=False)
    elif MODEL_SIGNATURE is not None:
        # MODEL_SIGNATURE may expect a tensor input; wrap numpy->tensor
        tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        try:
            outputs = MODEL_SIGNATURE(inputs=tensor)
            # outputs may be a dict-like mapping to tensors
            if isinstance(outputs, dict):
                # take the first tensor value as logits
                logits = list(outputs.values())[0]
            else:
                logits = outputs
        except Exception:
            # Some SavedModels accept a different keyword name; try common variants.
            try:
                outputs = MODEL_SIGNATURE(input_1=tensor)  # type: ignore
                logits = list(outputs.values())[0]
            except Exception as e:
                raise RuntimeError(f"Failed to call SavedModel signature: {e}")
    else:
        raise RuntimeError("No model loaded. Train and save a model first.")

    # logits is a Tensor; ensure it's in numpy for decoding
    logits_np = logits.numpy() if hasattr(logits, 'numpy') else np.array(logits)

    texts = greedy_ctc_decode(logits_np, VOCAB)
    return texts, tuple(logits_np.shape)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")

    arr = _preprocess_pil(img)

    try:
        texts, shape = predict_from_array(arr)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"filename": file.filename, "prediction": texts[0], "logits_shape": shape}


@app.get("/")
async def root():
    return RedirectResponse("http://localhost:5174")
