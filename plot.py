import pandas as pd
import pickle
import matplotlib.pyplot as plt

assessment: pd.DataFrame = pd.read_pickle('res50.pkl')

colors = ['red', 'green', 'blue', 'purple', 'orange']
for row, col in zip(assessment.itertuples(), colors):
    plt.plot(list(range(1, len(row.accuracy) + 1)), row.accuracy, ':', c=col)
    plt.plot(list(range(1, len(row.val_accuracy) + 1)), row.val_accuracy, '-', c=col, label=row.name)
plt.legend()
plt.xticks(list(range(1, len(row.accuracy) + 1)))
plt.suptitle('Pretrained ResNet: modifications')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.ylim((0,1))