from tensorflow.keras.backend import square
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Conv2D, Dropout, BatchNormalization, \
                         Reshape, Activation, Flatten, AveragePooling2D, MaxPooling2D, Conv3D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.constraints import max_norm

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy

import numpy as np

# mne imports
import mne
from mne import io
from mne.datasets import sample

# EEGNet-specific imports
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# MOABB imports
import moabb
from moabb.datasets import BNCI2014001, utils
from moabb.paradigms import MotorImagery
from moabb.evaluations import WithinSessionEvaluation
from moabb.evaluations import CrossSubjectEvaluation

# =====================================================================================
# EEGNet_8_2
# =====================================================================================

def EEGNet_8_2(nb_classes, chans, sp, loss='categorical_crossentropy', opt='adam', met=['accuracy']):
    CLASS_COUNT = nb_classes
    model = Sequential()    
    F1 = 8
    F2 = 16
    D = 2
    ks = 25 # kernel size 

    # Conv Block 1
    model.add(Conv2D(input_shape=(chans, sp, 1), filters=F1, kernel_size=(1, ks),
                        padding='same', use_bias = False))
    model.add(BatchNormalization())
    model.add(DepthwiseConv2D(kernel_size=(chans, 1), use_bias = False, 
                            depth_multiplier = D,
                            depthwise_constraint = max_norm(1.)))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(AveragePooling2D(pool_size=(1, 4)))
    model.add(Dropout(0.5))

    # Conv Block 2

    model.add(SeparableConv2D(filters=F2, kernel_size=(1, 16), padding = 'same', use_bias = False))
    model.add(BatchNormalization())
    model.add(Activation(activation='elu'))
    model.add(AveragePooling2D(pool_size=(1, 8)))
    model.add(Dropout(0.5))

    # Classification
    model.add(Flatten())
    model.add(Dense(CLASS_COUNT, kernel_constraint = max_norm(0.25)))
    model.add(Activation('softmax'))

    model.compile(loss=loss,
                optimizer=opt,
                metrics=met)
    
    return model


# =====================================================================================
# ShallowConvNet
# =====================================================================================

# need these for ShallowConvNet
def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000))

def ShallowConvNet(nb_classes, chans, sp, loss='categorical_crossentropy', opt='adam', met=['accuracy']):
    CLASS_COUNT = nb_classes
    model = Sequential()    

    # Conv Block 1
    model.add(Conv2D(input_shape=(chans, sp, 1), filters=40, kernel_size=(1, 13),
                        kernel_constraint = max_norm(2., axis=(0,1,2))))
    model.add(Conv2D(filters=40, kernel_size=(chans, 1), use_bias = False,
                        kernel_constraint = max_norm(2., axis=(0,1,2))))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(Activation(square))
    model.add(AveragePooling2D(pool_size=(1, 35), strides=(1, 7)))
    model.add(Activation(log))
    model.add(Dropout(0.5))

    # Classification
    model.add(Flatten())
    model.add(Dense(CLASS_COUNT, kernel_constraint = max_norm(0.5)))
    model.add(Activation('softmax'))

    model.compile(loss=loss,
                optimizer=opt,
                metrics=met)
    
    return model

# =====================================================================================
# DeepConvNet
# =====================================================================================

def DeepConvNet(nb_classes, chans, sp, loss='categorical_crossentropy', opt='adam', met=['accuracy']):
    CLASS_COUNT = nb_classes
    model = Sequential()    

    # Conv Block 1
    model.add(Conv2D(input_shape=(chans, sp, 1), filters=25, kernel_size=(1, 5), 
                        kernel_constraint = max_norm(2., axis=(0,1,2))))
    model.add(Conv2D(filters=25, kernel_size=(chans, 1),
                        kernel_constraint = max_norm(2., axis=(0,1,2))))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(Activation(activation='elu'))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
    model.add(Dropout(0.5))

    # Conv Block 2

    model.add(Conv2D(filters=50, kernel_size=(1, 5),
                        kernel_constraint = max_norm(2., axis=(0,1,2))))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(Activation(activation='elu'))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
    model.add(Dropout(0.5))
    
    # Conv Block 3

    model.add(Conv2D(filters=100, kernel_size=(1, 5),
                        kernel_constraint = max_norm(2., axis=(0,1,2))))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(Activation(activation='elu'))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
    model.add(Dropout(0.5))
    
    # Conv Block 4

    model.add(Conv2D(filters=200, kernel_size=(1, 5),
                        kernel_constraint = max_norm(2., axis=(0,1,2))))
    model.add(BatchNormalization(epsilon=1e-05, momentum=0.1))
    model.add(Activation(activation='elu'))
    model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2)))
    model.add(Dropout(0.5))

    # Classification
    model.add(Flatten())
    model.add(Dense(CLASS_COUNT, kernel_constraint = max_norm(0.5)))
    model.add(Activation('softmax'))

    model.compile(loss=loss,
                optimizer=opt,
                metrics=met)
    
    return model