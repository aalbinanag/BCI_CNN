# General imports
import os
import warnings
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import mne
import seaborn as sns
import pandas as pd

# Machine learning imports
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.pipeline import Pipeline

# MOABB imports
import moabb
from moabb.datasets import BNCI2014001, PhysionetMI, BNCI2014009, bi2013a
from moabb.paradigms import MotorImagery, P300
from moabb.evaluations import WithinSessionEvaluation, CrossSubjectEvaluation

# Local imports
import sys
sys.path.append('../../models') # go up two folders to reach the main folder containing 'models' folder
from models import EEGNet_8_2, ShallowConvNet, DeepConvNet

mne.set_log_level("CRITICAL")
moabb.set_log_level("info")
warnings.filterwarnings("ignore")

# Specify dataset and paradigm
dataset = BNCI2014009()
subj = [1, 2, 3, 4, 5, 6, 7, 8, 9]
dataset.subject_list = subj
nb_classes = 2
# paradigm = MotorImagery(nb_classes, fmin=4, fmax=40, tmin=0.5, tmax=2.5, resample= 128.0)
paradigm = P300(fmin=4, fmax=40, tmin=0.5, tmax=2.5, resample= 128.0)

## Obtain nb_channels and nb_samples for the input_shape of our model
X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[8])
channels = X.shape[1]
samples = X.shape[2]
batch_size = 16
epochs = 500
loss = 'categorical_crossentropy'
opt = 'adam'
met = ['accuracy']


# Define early stopping based on validation loss
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', monitor='val_loss', verbose=1,
                               save_best_only=True)

# Use KerasClassifier to wrap the Keras model within the Pipeline class
model = KerasClassifier(build_fn=lambda: EEGNet_8_2(nb_classes, channels, samples, loss=loss, opt=opt, met=met), 
                        epochs=epochs, 
                        batch_size=batch_size,
                        validation_split=0.2,
                        callbacks=[early_stopping, checkpointer],
                        verbose=1)

# Create pipeline
pipeline = Pipeline([('EEGNet_8_2', model)])                    

# Specify evaluation
evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=[dataset],
    overwrite=False,
    hdf5_path=None,
)

# Obtain results and store them as a csv
results = evaluation.process({"EEGNet_8_2": pipeline})
if not os.path.exists("./results"):
    os.mkdir("./results")
filename = f"results_benchmark_{os.path.splitext(os.path.basename(__file__))[0]}.csv"
results.to_csv(f"./results/{filename}")
results = pd.read_csv(f"./results/{filename}")
print(results)


