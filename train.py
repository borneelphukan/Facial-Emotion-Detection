from keras.layers import Activation, Convolution2D, Dropout
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from feature_extraction import load_dataset
from feature_extraction import input_processing
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

input_shape = (48,48,1)
patience = 50
batch_size = 32
n_epochs = 10000
num_classes = 7
verbose=1

gen_data = ImageDataGenerator(
    featurewise_center = False,
    featurewise_std_normalization = False,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.1,
    horizontal_flip = True)

model = Sequential()
model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                            name='image_array', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
model.add(Dropout(0.5))

model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3), padding='same'))
model.add(GlobalAveragePooling2D())
model.add(Activation('softmax',name='predictions'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#callbacks
csv_logger = CSVLogger('meta-data/training_log', append=False)
early_stop = EarlyStopping(monitor='val_loss', patience=patience)
new_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(patience/4), verbose=1)

saved_weights= ModelCheckpoint('meta-data/Weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
callbacks = [saved_weights, csv_logger, early_stop, new_lr]

#loading dataset
faces, emotions = load_dataset()
faces = input_processing(faces)
n_sample, num_classes = emotions.shape

x_train, x_test, y_train, y_test = train_test_split(faces, emotions, test_size = 0.2, shuffle = True)
steps_per_epoch = len(x_train)/batch_size
final_model = model.fit_generator(gen_data.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train)/batch_size, epochs = n_epochs, verbose = 1, callbacks = callbacks, validation_data = (x_test, y_test))

print(final_model.history.keys())

# Accuracy graph
plt.plot(final_model.history['accuracy'])
plt.plot(final_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Loss Graph
plt.plot(final_model.history['loss'])
plt.plot(final_model.history['val_loss'])
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
