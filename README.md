# ML_Garbage_Classification

Class project for CSCI 635-2 group 7


## Downloading Project Data

To download the project data, run the following script in the project's _code_ directory:
```
download_data.py
```

The script requires no command line arguments to run. However, it does require the user to have a
kaggle.com username and up to date api key. These two items must also be registered on the machine's
user variables, as follows:

```
KAGGLE_USERNAME: <your kaggle.com username>
```
```
KAGGLE_KEY: <your api key>
```
The script will download the files necessary, unzip them and install them to the data directory, and
report on the results. 



## Preprocessing Image Files

Included in the code is the preprocessing package. This package handles image processing tasks in preparation for model
training, validation and evaluation. The package contains four files: </br>
config.py</br>
transforms.py</br>
preprocess.py</br>
dataset.py</br>
</br>
### config.py
As its name suggests, config.py contains configuration data for the preprocessing module to tailor how preprocessing
tasks are completed and what options are included. DataConfig contains a number of fields for input/output options, as
well as options for how images are processed in the offline or online stages of preprocessing. The Classes object
simply contains the names of classes for the garbage classification task, as well as some basic property getters. The
options are as follows:
#### Directory location options -
```
raw_data_path
```
Where the preprocessing system will pull images from.
```
processed_data_path
```
Where the system will deposit preprocessed images for later training/testing/etc.
#### Image sizing options -
```
target_size
```
What size the preprocessing system will normalize all images to.
#### Training split options -
```
training_ratio
```
```
validation_ratio
```
```
test_ratio
```
These three values are split ratios for random division.
```
split_random_seed
```
Simply a numerical seed for randomization of selection.
#### CLAHE transform preprocessing options -
```
apply_clahe
```
A boolean flag to apply a CLAHE transform. This can help bring out smaller/less noticeable features without adding
noise.
```
clahe_clip_limit
```
An upper-bound threshold to cut out noise in the CLAHE transform.
```
clahe_tile_size
```
This reflects how big the CLAHE grid tiles are for the transform. Smaller tiles enhance smaller, more granular details.
Larger tiles help with feature extraction for broader, smoother contours.
#### Online (live) image augmentation options - 
```
apply_random_flip
```
A boolean flag that causes the system to randomly flip images. This is helpful to for reducing overfitting.
```
apply_random_rotation
```
A boolean flag to have the system randomly rotate the image one direction or another.
```
random_rotation_degrees
```
The maximum number of degrees the system will randomly rotate to clockwise or counterclockwise.
```
apply_color_jitter
```
A boolean flag to activate a color jitter in online augmentation. This helps to reduce overfitting.
```
color_jitter_brightness
```
This value controls the maximum brightness shift for the color jitter, if activated.
```
color_jitter_contrast
```
This value controls the maximum shift in contrast value, if the color jitter is activated.
```
color_jitter_saturation
```
This controls the maximum shift in saturation, if color jitter is activated.
```
color_jitter_hue
```
This value controls the maximum shift in hue, if color jitter is activated.
#### Normalization values -
```
normalization_mean
```
```
normalization_std
```
These values control the mean and standard deviation for the normalization the pixel values.
#### DataLoader values -
```
batch_size
```
Controls the number of images to work through in a single batch when processing.
```
num_workers
```
Controls the number of worker threads deployed concurrently.


### transforms.py
This file contains a set of file formatting and image transformation functions to be called in various preprocessing,
whether in the preprocessing stage, or ad-hoc for other forms of modelling.

### preprocess.py
Handles all offline image transformation and preprocessing steps. To run this part, simply call the filename. The
processed images are then deposited into the directory denoted in the config.

### dataset.py
This module handles all online transformations. It is capable of building datasets with a dataloader for each part of
the split and running the transformations while training. To make sure this part is compatible with local hardware, run
the file as is in terminal, which will call the "check_process" function and produce a sampling of images processed as
denoted in the config file.

## Running the training

After data is downloaded and preprocessed, ensure you are located in the 'code' directory and run:
```
python main.py --knn --lr
```
Base functionality will only train the CNN model

**Optional:** Including 'knn' will train the K-NN model

**Optional:** Including 'lr' will train the Logistic Regression model
