from keras.models import Sequential
from keras.layers.core import Dense , Dropout , Activation , Flatten
from keras.layers.convolutional import Convolution2D , MaxPooling2D
from keras.utils import np_utils
from keras.layers import Dense
from keras import optimizers
import keras
from keras.initializations import he_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano

# for importing image and then processing it.
#using PIL instead of opencv
from PIL import Image

from numpy import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import timeit
from keras import backend as K

K.set_image_dim_ordering('th')

start = timeit.default_timer()

#path of input data
path1 = "/Users/satvikkulshreshtha/Desktop/Pill_Classification/Pills"

#new folder for input in cnn
path2 = "/Users/satvikkulshreshtha/Desktop/Pill_Classification/Pills_resized"
img_rows = 227
img_cols = 227

listing = os.listdir(path1)
num_samples = size(listing)

print(num_samples)

# reading images from path1 and resized imaged grayscale saved to path2
for file in listing:
    im = Image.open(path1 + "/" + file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert("L")
    gray.save(path2 + "/" + file, "JPEG")

imlist = os.listdir(path2)
print(size(imlist))


#converting image to array for further manipulation
im1 = array(Image.open("Pills_resized" + "/" + imlist[0]))
m,n = im1.shape[0:2]
imnbr = len(imlist)
print(m) #row
print(n) #column

#putting all data in one matrix after flattening to feed in network
# samples, row/column (1291, 40000)
immatrix = array([array(Image.open("Pills_resized" + "/" + im2)).flatten()for im2 in imlist], "f")

#number of classes for output
#creating label for each class
label = np.ones((num_samples,), dtype=int)
label[0:708] = 0         #not round
label[708:808] = 1          #Polygon
label[808:] = 2          #Round

#shuffling the data before feeding it to network to make system stable and for good training
data,Label = shuffle(immatrix, label, random_state = 2)
train_data = [data, Label]

# just to check if there is any image in folder (shuffled)
'''
img = immatrix[167].reshape(img_rows, img_cols)
plt.imshow(img)
plt.show()
plt.imshow(img, cmap="gray")
plt.show()
'''
# train data[0] = data (1291, 40000)
print(train_data[0].shape)
#data[1] = labels (1291)
print(train_data[1].shape)

#Batch size to train
batch_size = 32  # batch size to be increased. atleast 100, actual was 32
#number of output classes
nb_classes = 3
#number or epochs to train
nb_epoch = 3
#input image dimension
img_rows, img_cols = 227, 227

#number of channels
img_channels = 1

# number of convolutional filters to use
nb_filters = 32

#size of pooling area for max pooling (pooling size)
nb_pool = 2

#convolutional kernel size (Window Size)
nb_conv = 3

#assigning variables to data
(X, y) = (train_data[0], train_data[1])

# Step 1 splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

#reshaping flattened matrix to 2D
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

#converting pixel value to float
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

#for faster computation
X_train /= 255
X_test /= 255

#printing
print("X_train shape:", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

#convert class vector to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#checking the final matrix image classified in labels
i = 100
plt.imshow(X_train[i, 0], interpolation="nearest")
plt.show()
print("label: ", Y_train[i,:])

#calling the model (similar to MNIST)
model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode = "valid", input_shape=(1, img_rows, img_cols)))
convout1 = Activation("relu")
model.add(convout1)
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
convout2 = Activation("relu")
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
convout3 = Activation("relu")
model.add(convout3)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
#model.add(Dropout(0.10))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation("relu"))
#model.add(Dropout(0.15))  # changed from 0.50
model.add(Dense(nb_classes)) # output layer
model.add(Activation("softmax"))

sgd = optimizers.SGD(lr=0.001, momentum=0.95, nesterov=True)
#sgd = optimizers.SGD(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#model.compile(loss="categorical_crossentropy", optimizer="adadelta",metrics=['accuracy'])
#model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])
#model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])

#model fitting and validation accuracy
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy = True, verbose=1, validation_data=(X_test, Y_test))
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy = True, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, Y_test, show_accuracy = True, verbose=0)
print("Test score: ", score[0])
print("Test accuracy: ", score[1])
stop = timeit.default_timer()
print('Runtime :', stop - start)
print(model.predict_classes(X_test[1:5]))
print(Y_test[1:5])

#Visualizing intermediate layer
#change model.layer[0 to 1] to see output of activation layer corrosponding to cnn
#output_layer = model.layers[0].get_output()
#output_function = theano.function([model.layers[0].get_input()], output_layer)

#the input image
input_image = X_train[0:1,:, :, : ]    #X_train [1 image, 1 channel, row, cols]
print(input_image.shape)
plt.imshow(input_image[0,0, :, :], cmap="gray")
#plt.show()
plt.imshow(input_image[0,0, :, :])
#plt.show()

#output_image = output_function(input_image)
#print(output_image.shape)

#rearrange the diemnsion so we can plot result as rgb image
#output_image = np.rollaxis(np.rollaxis(output_image, 3 , 1), 3 , 1)
#print(output_image.shape)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#Just to see what the code is learning
'''
fig = plt.figure(figsize=(8 , 8))
for i in range(32):
    ax= fig.add_subplot(6, 6, i+1)
    ax.imshow(output_image[0, :, :, i], cmap=matplotlib.cm.gray)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.tight_layout()
plt
'''

Y_pred = model.predict(X_test)
#print(Y_pred)
y_pred = np.argmax(Y_pred, axis=1)
#print(y_pred)

# saving weights

fname = "weights-Test-CNN.hdf5"
model.save_weights(fname,overwrite=True)


# Loading weights

fname = "weights-Test-CNN.hdf5"
model.load_weights(fname)

stop = timeit.default_timer()
print('Runtime :', stop - start)

