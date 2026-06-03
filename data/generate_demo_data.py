"""Generate a larger OCR demo dataset and a manifest.csv compatible with the project's loader.

Run from the project root with the venv activated:
  & .\.venv\Scripts\python.exe data\generate_demo_data.py
"""
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from pathlib import Path
import csv
import random

try:
        import numpy as np
except Exception:  # pragma: no cover - optional in the script runtime
        np = None

ROOT = Path(__file__).absolute().parent
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Generate a larger and more diverse set of base texts
NUM_BASE_SAMPLES = 200
VARIANTS_PER_SAMPLE = 3
SEED = 42

base_words = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "test",
    "sample",
    "OCR",
    "Document",
    "Invoice",
    "Total",
    "Name",
    "Address",
    "Date",
    "Reference",
    "Amount",
    "Value",
    "OpenAI",
    "Hello",
    "World",
    "Data",
]

def _make_random_text(rng: random.Random):
    words = [rng.choice(base_words) for _ in range(rng.randint(2, 6))]
    # add occasional numbers or punctuation
    if rng.random() < 0.3:
        words.append(str(rng.randint(1, 9999)))
    text = " ".join(words)
    return text

def make_image(path: Path, text: str, size=(256,256)):
    img = Image.new("RGB", size, color=(255,255,255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        # fallback
        text_w, text_h = font.getsize(text)
    x = (size[0] - text_w) // 2
    y = (size[1] - text_h) // 2
    draw.text((x,y), text, fill=(0,0,0), font=font)
    img.save(path)


def _add_noise(img: Image.Image, rng: random.Random, noise_level: int = 8) -> Image.Image:
    if np is None:
        return img

    arr = np.array(img).astype("int16")
    noise = np.array(
        [rng.randint(-noise_level, noise_level) for _ in range(arr.size)],
        dtype="int16",
    ).reshape(arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype("uint8")
    return Image.fromarray(arr)


def augment_image(base_img: Image.Image, rng: random.Random) -> Image.Image:
    img = base_img.copy()

    angle = rng.uniform(-4.0, 4.0)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255))

    shift_x = rng.randint(-6, 6)
    shift_y = rng.randint(-6, 6)
    img = img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, shift_x, 0, 1, shift_y),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )

    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.2))
    img = ImageEnhance.Sharpness(img).enhance(rng.uniform(0.8, 1.3))

    if rng.random() < 0.35:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.0)))

    if rng.random() < 0.7:
        img = _add_noise(img, rng)

    return img

def main():
    rng = random.Random(SEED)
    manifest_rows = []

    # create NUM_BASE_SAMPLES base images with unique texts
    for i in range(1, NUM_BASE_SAMPLES + 1):
        text = _make_random_text(rng)
        rel_path = f"images/img_{i:04d}.png"
        p = ROOT / rel_path
        make_image(p, text)
        manifest_rows.append((rel_path, text))

        base_img = Image.open(p).convert("RGB")
        stem = Path(rel_path).stem
        for variant_index in range(1, VARIANTS_PER_SAMPLE + 1):
            variant_name = f"{stem}_aug_{variant_index:02d}.png"
            variant_rel_path = f"images/{variant_name}"
            variant_path = ROOT / variant_rel_path
            variant_img = augment_image(base_img, rng)
            variant_img.save(variant_path)
            manifest_rows.append((variant_rel_path, text))

        base_img.close()

    manifest_path = ROOT / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "text"])
        for rel_path, text in manifest_rows:
            writer.writerow([rel_path, text])

    print(f"Wrote {len(manifest_rows)} images to {IMAGES_DIR} and manifest to {manifest_path}")

if __name__ == '__main__':
    main()
