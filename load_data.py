import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def load_train(top=50000, train_path='data/train', labels_path='data/trainLabels.csv', validation_train_ratio=0.3,
               rand_seed=None):

    top -= 1
    data = pd.read_csv(labels_path)[:top]
    images = [plt.imread(os.path.join(train_path, f"{i + 1}.png")) for i in range(top)]
    data['image'] = images
    np.random.seed(rand_seed)
    ind = np.random.choice(range(top), int(validation_train_ratio * top))
    train = data.loc[ind,].reset_index(drop=True)
    validation = data.drop(index=ind).reset_index(drop=True)
    return train, validation


def load_test(top=300000, train_path='data/test'):
    top -= 1
    images = pd.Series([plt.imread(os.path.join(train_path, f"{i + 1}.png")) for i in range(top)],
                       name='image').to_frame()
    return images
