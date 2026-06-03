import sys
from pathlib import Path
import tensorflow as tf
import numpy as np

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))

from ModelCreation.ViT import OCRVocabulary, ViTOCR, ctc_loss, greedy_ctc_decode
from DataManagment.OCRData import load_image

vocab = OCRVocabulary()
model = ViTOCR(vocab_size=vocab.size, image_size=256, patch_size=16)
model.build((None,256,256,3))
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# pick first two images
manifest = repo / 'data' / 'manifest.csv'
rows = manifest.read_text(encoding='utf-8-sig').strip().splitlines()[1:3]
images = []
labels = []
for r in rows:
    img_rel, text = r.split(',',1)
    img = load_image(str(repo / 'data' / img_rel), image_size=(256,256))
    images.append(img.numpy())
    labels.append(np.array(vocab.encode(text), dtype=np.int32))

images = tf.convert_to_tensor(np.stack(images, axis=0))
# pad labels to maxlen
maxlen = max(len(l) for l in labels)
labels_padded = np.zeros((len(labels), maxlen), dtype=np.int32)
for i,l in enumerate(labels):
    labels_padded[i,:len(l)] = l
labels = tf.convert_to_tensor(labels_padded)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = tf.reduce_mean(ctc_loss(labels, logits))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, logits

for step in range(1,201):
    loss, logits = train_step(images, labels)
    if step % 20 == 0:
        decoded = greedy_ctc_decode(logits.numpy(), vocab)
        print(f"step={step} loss={float(loss.numpy()):.4f} decoded={decoded}")

print('done')
