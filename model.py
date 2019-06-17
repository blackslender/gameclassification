import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Conv2D, Activation, Add, MaxPooling2D, BatchNormalization, Dense, Input, ZeroPadding2D
from keras.models import Model
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

path = ""
training_data_dir = path + "training"  # 10 000 * 2
validation_data_dir = path + "validating"  # 2 500 * 2
test_data_dir = path + "testing"  # 12 500
IMAGE_HEIGHT = 90
IMAGE_WIDTH = 160
BATCH_SIZE = 10
EPOCHS = 500
num_classes = 17


###### Model: Resnet ######
def identityBlock(f, filters, stage, block):
    def func(x):
        # Defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters
        X = x
        # Save the input value
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters=F1,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding='valid',
                   name=conv_name_base + '2a',
                   kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('selu')(X)

        # Second component of main path
        X = Conv2D(filters=F2,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding='valid',
                   name=conv_name_base + '2b',
                   kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('selu')(X)

        # Third component of main path
        X = Conv2D(filters=F3,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding='valid',
                   name=conv_name_base + '2c',
                   kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
        X = Activation('selu')(X)

        # Final step: Add shortcut value to main path and pass it through a selu activation
        X = Add()([X, X_shortcut])
        X = Activation('selu')(X)
        return X
    return func


def convolutionalBlock(f, filters, stage, block, s=2):
    def func(x):
        # Defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrive Filters
        F1, F2, F3 = filters
        X = x
        # Save the input value
        X_shortcut = X

        ### MAIN PATH ###

        # First component of main path
        X = Conv2D(filters=F1,
                   kernel_size=(1, 1),
                   strides=(s, s),
                   padding='valid',
                   name=conv_name_base + '2a',
                   kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('selu')(X)

        # Second component of main path
        X = Conv2D(filters=F2,
                   kernel_size=(f, f),
                   strides=(1, 1),
                   padding='same',
                   name=conv_name_base + '2b',
                   kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('selu')(X)

        # Third component of main path
        X = Conv2D(filters=F3,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   padding='valid',
                   name=conv_name_base + '2c',
                   kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
        X = Activation('selu')(X)

        ### SHORTCUT PATH ###
        X_shortcut = Conv2D(filters=F3,
                            kernel_size=(1, 1),
                            strides=(s, s),
                            padding='valid',
                            name=conv_name_base + '1',
                            kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(
            axis=3, name=bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path. and pass it through a selu activation
        X = Add()([X, X_shortcut])
        X = Activation('selu')(X)

        return X
    return func


def resnet50(input_shape=(64, 64, 3), classes=6):

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), name='conv1',
               kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('selu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutionalBlock(
        f=3, filters=[64, 64, 256],   stage=2, block='a', s=1)(X)
    X = identityBlock(3, [64, 64, 256], stage=2, block='b')(X)
    X = identityBlock(3, [64, 64, 256], stage=2, block='c')(X)

    # Stage 3
    X = convolutionalBlock(
        f=3, filters=[128, 128, 512], stage=3, block='a', s=2)(X)
    X = identityBlock(3, [128, 128, 512], stage=3, block='b')(X)
    X = identityBlock(3, [128, 128, 512], stage=3, block='c')(X)
    X = identityBlock(3, [128, 128, 512], stage=3, block='d')(X)

    # Stage 4
    X = convolutionalBlock(
        f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)(X)
    X = identityBlock(3, [256, 256, 1024], stage=4, block='b')(X)
    X = identityBlock(3, [256, 256, 1024], stage=4, block='c')(X)
    X = identityBlock(3, [256, 256, 1024], stage=4, block='d')(X)
    X = identityBlock(3, [256, 256, 1024], stage=4, block='e')(X)
    X = identityBlock(3, [256, 256, 1024], stage=4, block='f')(X)

    # Stage 5
    X = convolutionalBlock(
        f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)(X)
    X = identityBlock(3, [512, 512, 2048], stage=5, block='b')(X)
    X = identityBlock(3, [512, 512, 2048], stage=5, block='c')(X)

    # AVGPOOL
    X = MaxPooling2D(pool_size=(2, 2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes),
              kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    # The model should be compiled before training
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


model = resnet50(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
                 classes=num_classes)

###########################
training_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)
validation_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)

training_generator = training_data_generator.flow_from_directory(
    training_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE
)
validation_generator = validation_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE
)
test_generator = test_data_generator.flow_from_directory(
    test_data_dir,
    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=1,
    shuffle=False)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# define the checkpoint
filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
csvLogger = CSVLogger('log',append = False, separator = ';')
callbacks_list = [checkpoint,earlyStopping,csvLogger]

model.fit_generator(
    training_generator,
    steps_per_epoch=len(training_generator.filenames),
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(validation_generator.filenames),
    callbacks = callbacks_list)
#  // BATCH_SIZE,
# callbacks=[PlotLossesKeras(), CSVLogger(TRAINING_LOGS_FILE,
#                                         append=False,
#                                         separator=";")])
