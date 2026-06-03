from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import sys

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from ModelCreation.ViT import OCRVocabulary, ViTOCR, greedy_ctc_decode
from DataManagment.OCRData import load_image

MANIFEST = repo / 'data' / 'manifest.csv'
# pick first image
with MANIFEST.open('r', encoding='utf-8-sig') as f:
    header = f.readline()
    first = f.readline().strip().split(',',1)
    img_rel = first[0]
    text = first[1]

img_path = repo / 'data' / img_rel
print('Sample:', img_rel, text)

vocab = OCRVocabulary()
model = ViTOCR(vocab_size=vocab.size, image_size=256, patch_size=16)
# build
model.build((None,256,256,3))

img = load_image(str(img_path), image_size=(256,256))
img = tf.expand_dims(img, 0)

logits = model(img, training=False)
print('Logits shape:', logits.shape)

argmax = tf.argmax(logits, axis=-1).numpy()
print('Argmax sample (first time-steps):', argmax[0][:30])

texts = greedy_ctc_decode(logits.numpy(), vocab)
print('Decoded:', repr(texts[0]))

# print softmax max probs distribution
probs = tf.nn.softmax(logits, axis=-1).numpy()
maxp = probs.max(axis=-1)
print('Max prob per timestep (first 30):', maxp[0][:30])

# show proportion of blank token predictions
blank_id = vocab.blank_id
blank_prop = (argmax == blank_id).sum() / argmax.size
print('Blank proportion:', blank_prop)
