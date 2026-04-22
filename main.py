import numpy
import cv2
import pandas
import seaborn
import matplotlib.pyplot as plt
import tensorflow
import tensorflow.keras
from sklearn.model_selection import train_test_split


from ModelCreation.ViT import ViT


encoder = ViT().config(shape=(128,128), patch_size=20)



