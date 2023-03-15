# General imports
import os
import warnings
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import mne
from mne.decoding import CSP
import seaborn as sns

# Machine learning imports
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import pandas as pd
from joblib import dump, load
import pickle


# MOABB imports
import moabb
from moabb.datasets import BNCI2014001
from moabb.evaluations import CrossSessionEvaluation, WithinSessionEvaluation
from moabb.pipelines.utils import FilterBank


# Local imports
from utils.models.EEGNet_v2 import model_EEGNet
from utils.models.shallow_cnn import Shallow_CNN
from utils.models.EEGNet3 import EEGNet
from utils.utils_2 import Estimator
from utils.local_paradigms import LeftRightImageryAccuracy,FilterBankLeftRightImageryAccuracy


mne.set_log_level("CRITICAL")
moabb.set_log_level("info")
warnings.filterwarnings("ignore")
    
print(tensorflow.__version__)


#############################################################################
# MOABB application

## Parameters
classes = 2
channels = 22 #defined by dataset
sp = 1001 #defined by dataset
batch_size = 64
epochs = 1
lr = 0.01
loss = 'sparse_categorical_crossentropy' #categorical_crossentropy
opt = Adam(lr=lr)
met = ['accuracy'] #auc(for roc_auc metrics used for two classes in MOABB) #accuracy

## Making pipelines
print("Making pipelines ")
pipelines={}
clf_shallow_cnn = Shallow_CNN('shallow_cnn')
clf_shallow_cnn = clf_shallow_cnn.create_model(classes, loss=loss, opt=opt, met=met)
pipe = make_pipeline(Estimator(clf_shallow_cnn,'shallow_cnn', batch_size))
pipelines['shallow_cnn'] = pipe

# clf_EEGNet = model_EEGNet(classes, channels, sp, epochs, loss, opt, met)
# pipe = make_pipeline(Estimator(clf_EEGNet, 'EEGNet', batch_size))
# pipelines['EEGNet'] = pipe

clf_EEGNet = EEGNet('EEGNet')
clf_EEGNet = clf_EEGNet.create_model(classes, loss=loss, opt=opt, met=met)
pipe = make_pipeline(Estimator(clf_EEGNet,'EEGNet', batch_size))
pipelines['EEGNet'] = pipe


## Specifying datasets, paradigm and evaluation
print("Specifying datasets, paradigms and evaluation ")
dataset = BNCI2014001()

paradigm = LeftRightImageryAccuracy() #2 classes (right and left hands) with accuracy metric
evaluation = CrossSessionEvaluation(paradigm=paradigm, datasets=dataset, overwrite=False)

## Specifying datasets, paradigm and evaluation
print("Specifying datasets, paradigms and evaluation ")
datasets = [BNCI2014001()]
paradigm = LeftRightImageryAccuracy() #2 classes (right and left hands) with accuracy metric
evaluation = CrossSessionEvaluation(paradigm=paradigm, datasets=datasets, overwrite=False)

## Getting and saving results
print("Calculating results ")
results = evaluation.process(pipelines)
if not os.path.exists("./results"):
    os.mkdir("./results")
results.to_csv("./results/results_benchmark.csv")
results = pd.read_csv("./results/results_benchmark.csv")



