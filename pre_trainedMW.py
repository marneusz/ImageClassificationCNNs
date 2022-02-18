import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

from load_data import load_data_keras

x_train, x_test, y_train, y_test = load_data_keras()

initializer = keras.initializers.he_normal()

input_tensor = keras.Input(shape=(32, 32, 3))

# https://www.mathworks.com/help/deeplearning/ref/densenet201.html

resized_images = keras.layers.Lambda(lambda image: tf.image.resize(image, (224, 224)))(input_tensor)
model = keras.applications.DenseNet201(include_top=False,
                                       weights='imagenet',
                                       input_tensor=resized_images,
                                       input_shape=(224, 224, 3),
                                       pooling='max',
                                       classes=1000)
for layer in model.layers:
    layer.trainable = False

output = model.layers[-1].output

flatten = keras.layers.Flatten()
output = flatten(output)

layer_256 = keras.layers.Dense(units=256,
                               activation='elu',
                               kernel_initializer=initializer,
                               kernel_regularizer=keras.regularizers.l2())
output = layer_256(output)
dropout = keras.layers.Dropout(0.5)
output = dropout(output)
softmax = keras.layers.Dense(units=10,
                             activation='softmax',
                             kernel_initializer=initializer,
                             kernel_regularizer=keras.regularizers.l2())
output = softmax(output)
model = keras.models.Model(inputs=input_tensor, outputs=output)

model.compile(
         optimizer=keras.optimizers.Adam(learning_rate=1e-4),
         loss='categorical_crossentropy',
         metrics=['accuracy'])

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                                       initial_learning_rate=1e-4,
                                       decay_steps=10000,
                                       decay_rate=0.9)

# reduce learning rate when val_accuracy has stopped improving
lr_reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                              factor=0.6,
                                              patience=2,
                                              verbose=1,
                                              mode='max',
                                              min_lr=1e-7)
# stop training when val_accuracy has stopped improving
early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                           patience=3,
                                           verbose=1,
                                           mode='max')

# callback to save the Keras model and (best) weights obtained on an epoch basis. here,
# the trained (compiled) model is saved in the current working directory as 'cifar10.h5'
checkpoint = keras.callbacks.ModelCheckpoint('cifar10.h5',
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_weights_only=False,
                                             save_best_only=True,
                                             mode='max',
                                             save_freq='epoch')

train_datagen = keras.preprocessing.image.ImageDataGenerator(
                                          #rotation_range=20,
                                          # width_shift_range=0.2,
                                          # height_shift_range=0.2,
                                          # shear_range=0.2,
                                          zoom_range=0.1,
                                          horizontal_flip=True,
                                          #fill_mode='nearest'
)

train_generator = train_datagen.flow(x_train,
                                     y_train,
                                     batch_size=32)
val_datagen = keras.preprocessing.image.ImageDataGenerator(
                                        # rotation_range=40,
                                        # width_shift_range=0.2,
                                        # height_shift_range=0.2,
                                        # shear_range=0.2,
                                        zoom_range=0.1,
                                        horizontal_flip=True,
                                        # fill_mode='nearest')
)

val_generator = val_datagen.flow(x_test,
                                 y_test,
                                 batch_size=32)

train_steps_per_epoch = x_train.shape[0] // 32
val_steps_per_epoch = x_test.shape[0] // 32

history = model.fit(train_generator,
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=val_steps_per_epoch,
                    epochs=20,
                    shuffle=True,
                    callbacks=[lr_reduce, early_stop, checkpoint],
                    verbose=1)
