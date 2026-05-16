"""tutaj dalem tarkie testowe zeby
sprawdzic czy w oogle dziala --- dziala"""

from ViT import ViT
import tensorflow as tf

model = ViT(num_classes=36)

dummy = tf.random.normal((2, 256, 256, 3))

output = model(dummy)

print(output.shape)
