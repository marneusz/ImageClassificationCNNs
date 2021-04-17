from keras import Sequential
from sklearn.model_selection import train_test_split
from keras.applications import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3)

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

train_generator = ImageDataGenerator(
    rotation_range=2,
    horizontal_flip=True,
    zoom_range=.1)

val_generator = ImageDataGenerator(
    rotation_range=2,
    horizontal_flip=True,
    zoom_range=.1)

test_generator = ImageDataGenerator(
    rotation_range=2,
    horizontal_flip=True,
    zoom_range=.1)
train_generator.fit(x_train)
val_generator.fit(x_val)
test_generator.fit(x_test)

lrr = ReduceLROnPlateau(
    monitor='val_acc',
    factor=.01,
    patience=3,
    min_lr=1e-5)

base_model_2 = ResNet50(include_top=False, weights='imagenet', input_shape=x_train.shape[1:], classes=y_train.shape[1])

batch_size = 100
epochs = 50
learn_rate = .001

sgd = SGD(lr=learn_rate, momentum=.9, nesterov=False)
adam = Adam(lr=learn_rate, beta_1=0.9, beta=0.999, epsilon=None, decay=0.0, amsgrad=False)
model = Sequential()
# Add the Dense layers along with activation and batch normalization
model.add(base_model)
model.add(Flatten())

# Add the Dense layers along with activation and batch normalization
model.add(Dense(4000, activation=('relu'), input_dim=512))
model.add(Dense(2000, activation=('relu')))
model.add(Dropout(.4))
model.add(Dense(1000, activation=('relu')))
model.add(Dropout(.3))  # Adding a dropout layer that will randomly drop 30% of the weights
model.add(Dense(500, activation=('relu')))
model.add(Dropout(.2))
model.add(Dense(10, activation=('softmax')))  # This is the classification layer

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_generator.flow(x_train, y_train, batch_size=batch_size),
                    epochs=100, steps_per_epoch=x_train.shape[0] // batch_size,
                    validation_data=val_generator.flow(x_val, y_val, batch_size=batch_size), validation_steps=250,
                    callbacks=[lrr], verbose=1)
