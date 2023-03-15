from .model import KerasModel
from tensorflow.keras.backend import square
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Conv2D, Dropout, BatchNormalization, \
                         Reshape, Activation, Flatten, AveragePooling2D, Conv3D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.constraints import max_norm

class EEGNet(KerasModel):

    def create_model(self, nb_classes, augmented_data=False, print_summary=False, downsampled=False, loss='categorical_crossentropy', opt='adam', met=['accuracy']):

        CLASS_COUNT = nb_classes
        model = Sequential()
        # augmented_data = False
        # print_summary=False

        chans = 22 #3
        sp = 1001 #512
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
        model.add(Dense(CLASS_COUNT))
        model.add(Activation('softmax'))

        print(model.summary())

        # compile the model
        print("Compiling model...")
        model.compile(loss=loss,
                      optimizer=opt,
                      metrics=met)
        print("Compiling finsihed")

        # assign and return
        self.model = model
        return model
