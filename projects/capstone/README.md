# Machine Learning Engineer Nanodegree
## Specializations
## Project: Capstone Proposal and Capstone Project

**Note**

The Capstone is a two-staged project. The first is the proposal component, where you can receive valuable feedback about your project idea, design, and proposed solution. This must be completed prior to your implementation and submitting for the capstone project. 

You can find the [capstone proposal rubric here](https://review.udacity.com/#!/rubrics/410/view), and the [capstone project rubric here](https://review.udacity.com/#!/rubrics/108/view). Please ensure that you are following directions correctly before submitting these two stages which encapsulate your capstone.

Please email [machine-support@udacity.com](mailto:machine-support@udacity.com) if you have any questions.

## Software / Libraries

The capstone project uses Python and Python based toolkits for creating and training the models.  The code written in `capstone-model-engine.py` is the main python file that loades the images, performs pre-processing, defines the models that were used in training experiments, and trains the models.  

The following tools / libraries should be installed in order to run the `capstone-model-engine.py` file:

- Language: `Python 2.7` or `Python 3.5+` - https://www.python.org/downloads/
- Install tools: `pip` - https://pip.pypa.io/en/stable/installing/  OR `conda` - https://conda.io/docs/user-guide/install/index.html


- `tensorflow_gpu` - following instructions at https://www.tensorflow.org/install/ for the appropriate platform.
- `keras` - Neural Network API - https://keras.io/#installation
- `numpy` - Python Numerical analysis package - https://matplotlib.org/users/installing.html#installing-an-official-release
- `scipy` - Python packages - https://www.scipy.org/install.html
- `panda` - Python data analysis library - http://pandas.pydata.org/
- `matplotlib` - Python 2D plotting library - https://matplotlib.org/users/installing.html#installing-an-official-release
- `opencv` - Computer vision library - https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_setup/py_table_of_contents_setup/py_table_of_contents_setup.html
- `Pillow` - Python Imaging library - https://pillow.readthedocs.io/en/3.1.x/installation.html
- `scikit-learn` - Python based machine learning libraries - http://scikit-learn.org/stable/install.html
- `tqdm` - progress meters - https://pypi.python.org/pypi/tqdm
- `ipython` - Used to create images of the models - https://ipython.org/ipython-doc/2/install/install.html
- `Jupyter` - Python notebooks  - https://jupyter.readthedocs.io/en/latest/install.html

The following is **only required** if NVIDIA GPU needs to be used for faster computations, otherwise, install the appropriate drivers/libraries for the specific GPU being used.

- NVIDIA cuDNN 7.0 - Deep Neural Network Library - http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html

- NVIDIA CUDA 9.0 - https://developer.nvidia.com/cuda-90-download-archive

- NVIDIA Drivers - http://www.nvidia.com/Download/index.aspx

  ​

## Running the Program

#### Pre-requisites

1. The image data is provided in `sample_images/train` folder with `c0` to `c9` folders contains images for each class.

2. `capstone-model-engine.py` is the main class that will need to be executed.  

3. The following parameters are configurable in the `capstone-model-engine.py` file:

   - ```python
     #location to save models
     file_root='saved_models/'
     # prefix used for saving the model and history files
     prefix_str = str(datetime.date.today()) + str(random.randint(1, 100))

     # Number of epochs to perform training for
     NUM_EPOCHS = 1
     ImageFile.LOAD_TRUNCATED_IMAGES = True

     #Location of the image files
     images_path = "sample_images/train"
     ```

4. *Optional:* Appropriate GPU drivers are installed such as NVIDIA CUDA / cuDNN system libraries.

#### Executing the Model training 

1. Activate Tensorflow

```shell
> activate tensorflow
```

2. At the *tensorflow* prompt, run the model training program - *redirecting to a log file is optional*

```
(tensorflow) >python capstone-model-engine.py > logs/my.log
```



#### Model Training Output

The following files are generated in the `saved_models` folder:

- `create_base_model<date><random#>_complete_model.hdf5` - model architecture and weights.  This can be loaded using *Keras* `load_model()` function.
- `create_base_model<date><random#>_trainHistoryDict`  - The model history is stored in a pickle file
- `create_base_model<date><random#>_model_accuracy.png`  - The graph of the model training / validation curves.
- `create_base_model<date><random#>_model_loss.png` - The graph of the model training/ validation loss curves.



## References to supporting materials

The images used were from the distracted driver Kaggle competition sponsored by State Farm.  The project page is available at https://www.kaggle.com/c/state-farm-distracted-driver-detection.

The images are available at https://www.kaggle.com/c/5048/download/imgs.zip

The images are provided in the `train` folder that contains images for each class in the appropriate subfolder, and the `test` folder that contains a large set of unlabeled images that can be used for testing out the models.  Only a sample of images were used from the `test` folder to manual identify the image class and compare with the model predictions.



