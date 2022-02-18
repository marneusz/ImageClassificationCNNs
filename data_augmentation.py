import pandas as pd
import numpy as np
import cv2

from copy import deepcopy
from skimage import transform, util
from load_data import load_train, load_data_keras
import tensorflow as tf
tf.get_logger().setLevel('INFO')


def horizontal_flip(image_arr: np.array):
    return image_arr[:, ::-1]


def random_rotation(image_arr: np.array):
    random_angle = np.random.uniform(-15, 15)
    return transform.rotate(image_arr, random_angle)


def random_noise(image_arr: np.array):
    return util.random_noise(image_arr)


def zoom_image(image_arr: np.array, zoom: float = 3.0):
    img_size = image_arr.shape
    return tf.image.random_crop(image_arr, size=img_size)


available_transformations = {
    'rotation': random_rotation,
    'noise': random_noise,
    'horizontal': horizontal_flip,
    'zoom': zoom_image
}


def generate_data(data: pd.DataFrame, num_new: int = 100):
    num_generated = 0
    n_rows = data.shape[0]
    while num_generated < num_new:
        img_number = np.random.randint(0, n_rows-1)
        img_to_transform = deepcopy(data.iloc[img_number])
        transformation = np.random.choice(list(available_transformations))
        img_to_transform["image"] = available_transformations[transformation](img_to_transform["image"])
        data = data.append(img_to_transform, ignore_index=True)
        num_generated += 1
    return data


def show_img(image_arr: np.array, label):
    image = cv2.resize(image_arr, (400, 400))
    cv2.imshow(label, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def tests():
    X_train, _ = load_train(top=100, validation_train_ratio=1)
    print(X_train.shape)
    X_train = generate_data(X_train, 10)
    print(X_train.shape)


if __name__ == "__main__":
    tests()
