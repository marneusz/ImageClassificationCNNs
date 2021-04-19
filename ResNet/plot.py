import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def average_two_lists(a, b):
    '''Compute average of two lists of different length'''
    length = min(len(a), len(b))
    a = np.array(a)
    b = np.array(b)
    result = (a[:length] + b[:length]) / 2
    if len(a) < len(b):
        result = np.concatenate([result, b[length:]])
    else:
        result = np.concatenate([result, a[length:]])
    return result


assessment0 = pd.read_pickle('data/assessment_0.pkl')
assessment1 = pd.read_pickle('data/assessment_1.pkl')
colors = ['red', 'green', 'blue', 'purple', 'orange']
for row, col in enumerate(colors):
    accuracy = average_two_lists(assessment0.loc[row, 'accuracy'], assessment1.loc[row, 'accuracy'])
    val_accuracy = average_two_lists(assessment0.loc[row, 'val_accuracy'], assessment1.loc[row, 'val_accuracy'])
    plt.plot(list(range(1, len(accuracy) + 1)), accuracy, ':', c=col)
    plt.plot(list(range(1, len(val_accuracy) + 1)), val_accuracy, '-', c=col, label=assessment0.loc[row, 'name'])
plt.legend()
# plt.xticks(plt.xticks()[0].astype(int))
plt.title('Dotted line: train accuracy, \nSolid line: validation accuracy \nAveraged over two experiments')
plt.suptitle('Pretrained ResNet: modifications', fontsize=20)
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.ylim((0.5, 1))
