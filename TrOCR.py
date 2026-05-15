"""
TrOCR -> Vision Transformer + text decoder made by me - DanySwag
"""

from transformers import ViTImageProcessor
from transformers import VisionEncoderDecoderModel
from PIL import Image
import requests

processor = TrOCRProcessor.from_pretrained(
    "microsoft/trocr-base-printed"
)

model = VisionEncoderDecoderModel.from_pretrained(
    "microsoft/trocr-base-printed"
)

image = Image.open("example_document.jpg").convert("RGB")