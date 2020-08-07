from HodaDatasetReader import read_hoda_cdb, read_hoda_dataset
import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop


# In case you have problem running tensorflow 2 on gpu
tf.config.experimental.set_visible_devices([], 'GPU')


print('\n ****************************************************************************** \n')

# Reading Persian digit images dataset

print('Reading train dataset (Train 60000.cdb)...')
X_train, Y_train = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
                                images_height=28,
                                images_width=28,
                                one_hot=True,
                                reshape=False)


print('Reading test dataset (Test 20000.cdb)...')
X_test, Y_test = read_hoda_dataset(dataset_path='./DigitDB/Test 20000.cdb',
                              images_height=28,
                              images_width=28,
                              one_hot=True,
                              reshape=False)


print("\n ****************************************************************************** \n")

# Reading English digit images dataset

print("reading mnist data from keras dataset")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = x_train.reshape(60000, 28, 28, 1)
y_test = keras.utils.to_categorical(y_test, 10)
y_train = keras.utils.to_categorical(y_train, 10)

# Setting model's optimizer parameters

opt = optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Function to build model structure

def build_primary_model():

    # The Sequential model is a linear stack of layers that you can add layers via add() method
    model = Sequential()

    # 2D convolution layer. This layer creates a convolution kernel that is convolved
    # with the layer input to produce a tensor of outputs
    # 64 convolution filters of size 3x3 each . input image with size 28 * 28 and 1 depth for black and white
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

    # Downsampling layer . pool_size : factors by which to downscale (vertical, horizontal)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattens the input
    model.add(Flatten())

    return model

# ******************************************************************************

# Defining train parameters

# The number of samples that will be propagated through the network
# networks train faster with mini-batches. That's because we update the weights after each propagation
batch_size = 128

# Number of output classes
num_classes = 10

# One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network
epochs = 12

# Implementing base model to use for both Persian and English dataset training
base_model = build_primary_model()
model_for_Per_Eng = Sequential()
model_for_Per_Eng.add(base_model)

# Creating a hidden fully-connected layer, with 128 nodes.
model_for_Per_Eng.add(Dense(128, activation='relu'))

# This layer drops out a random set of activations in that layer by setting them to zero
# Fraction of the input units to drop
model_for_Per_Eng.add(Dropout(0.3))

model_for_Per_Eng.add(Dense(num_classes, activation='softmax'))

# Configure the learning process before training a model
# for multi-class classification problem.
# loss : loss function. This is the objective that the model will try to minimize
# For any classification problem set metrics=['accuracy']

model_for_Per_Eng.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
model_for_Per_Eng.build()
model_for_Per_Eng.summary()

model_for_Per_Eng.fit(X_train, Y_train, batch_size=128, epochs=12, verbose=1, validation_split=0.1)


score_on_Persian = model_for_Per_Eng.evaluate(X_test, Y_test)
score_on_English = model_for_Per_Eng.evaluate(x_test, y_test)

base_model.save_weights('modelforEngPer.h5')

print('Test loss:', score_on_Persian[0])
print('Test accuracy:', score_on_Persian[1])
print('Test loss English:', score_on_English[0])
print('Test accuracy English:', score_on_English[1])

# ******************************************************************************

# Using saved weights from previous training and tuning final layers to improve English dataset accuracy

basemodel_for_Eng_WithoutTrainFE_Layer = build_primary_model()
basemodel_for_Eng_WithoutTrainFE_Layer.trainable = False
basemodel_for_Eng_WithoutTrainFE_Layer.load_weights('modelforEngPer.h5')

model_for_Eng_WithoutTrainFE_Layer = Sequential()
model_for_Eng_WithoutTrainFE_Layer.add(basemodel_for_Eng_WithoutTrainFE_Layer)

model_for_Eng_WithoutTrainFE_Layer.add(Dense(64, activation='relu'))
model_for_Eng_WithoutTrainFE_Layer.add(Dropout(0.3))
model_for_Eng_WithoutTrainFE_Layer.add(Dense(num_classes, activation='softmax'))
model_for_Eng_WithoutTrainFE_Layer.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
model_for_Eng_WithoutTrainFE_Layer.summary()
model_for_Eng_WithoutTrainFE_Layer.fit(x_train, y_train, batch_size=128, epochs=12, verbose=1)
score_on_English_WithoutTraining = model_for_Eng_WithoutTrainFE_Layer.evaluate(x_test, y_test)
print('English Without Training loss:', score_on_English_WithoutTraining[0])
print('English Without Training accuracy:', score_on_English_WithoutTraining[1])


