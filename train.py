from cnn import xceptionNetwork
from feature_extraction import load_data, preprocessing
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

batch_size = 32
n_epochs = 1000
input_shape = (48,48,1)
validation_split = 0.2
num_classes = 7

data_generator = ImageDataGenerator(featurewise_center=False, rotation_range=10, featurewise_std_normalization=False, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)
model = xceptionNetwork(input_shape, num_classes)

log_generator = 'training.log'
csv_logger = CSVLogger(log_generator, append=False)
early_stop = EarlyStopping('val_loss', patience=50)
lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(50/4), verbose=1)

model_names = 'alexnet.hdf5'
checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1, save_best_only=True)
callbacks = [checkpoint, csv_logger, early_stop, lr]

faces, face_emotions = load_data()
faces = preprocessing(faces)
num_samples, num_classes = face_emotions.shape
x_train, x_test, y_train, y_test = train_test_split(faces, face_emotions, test_size = 0.2, shuffle = True)
model.fit_generator(data_generator.flow(x_train, y_train, batch_size), steps_per_epoch=len(x_train)/batch_size, epochs=n_epochs, verbose=1, callbacks=callbacks, validation_data=(x_test, y_test))