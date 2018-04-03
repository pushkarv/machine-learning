
import glob
import tensorflow as tf
from keras.models import load_model

from keras.utils import plot_model
model_files = glob.glob('final_model/create*.hdf5')
for m in model_files:
    print("Loading model: ", f)
    model = load_model(f)
    

