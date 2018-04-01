
import glob
from keras.models import load_model

from keras.utils import plot_model
model_files = glob.glob('saved_models/create*.hdf5')
for m in model_files:
    f = m.split('/')[1]
    print(f)
    print("Loading model: ", f)
    model = load_model('saved_models/' + f)
    plot_model(model, to_file='./model_pics/' + f.split('.')[0] + '.png', show_shapes=True, show_layer_names=True)

