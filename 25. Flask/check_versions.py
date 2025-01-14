import importlib.metadata
import keras
import tensorflow as tf
import flask
import numpy as np
import pandas as pd
from PIL import Image
# import gunicorn

print("Keras version:", keras.__version__)
print("TensorFlow version:", tf.__version__)
print("Flask version:", importlib.metadata.version("flask"))
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Pillow version:", Image.__version__)
# print("Gunicorn version:", gunicorn.__version__)