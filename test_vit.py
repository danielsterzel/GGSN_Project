import sys
sys.path.append('.')
from ModelCreation import ViT
import numpy as np

print('Import OK')
v = ViT.ViTOCR(vocab_size=100, image_size=128, patch_size=16)
print('num_patches (init):', v.encoder.num_patches)
fake = np.random.rand(1, 128, 128, 3).astype('float32')
logits = v(fake)
print('logits shape', logits.shape)
