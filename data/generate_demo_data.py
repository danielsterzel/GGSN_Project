"""Generate small demo images and a manifest.csv compatible with the project's loader.

Run from the project root with the venv activated:
  & .\.venv\Scripts\python.exe data\generate_demo_data.py
"""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import csv

ROOT = Path(__file__).absolute().parent
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

samples = [
    ("images/img_001.png", "Hello"),
    ("images/img_002.png", "Test OCR"),
    ("images/img_003.png", "123 ABC"),
    ("images/img_004.png", "OpenAI"),
    ("images/img_005.png", "Sample Text"),
    ("images/img_006.png", "Line two"),
]

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

def main():
    for rel_path, text in samples:
        p = ROOT / rel_path
        make_image(p, text)

    manifest_path = ROOT / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_path", "text"])
        for rel_path, text in samples:
            writer.writerow([rel_path, text])

    print(f"Wrote {len(samples)} images to {IMAGES_DIR} and manifest to {manifest_path}")

if __name__ == '__main__':
    main()
