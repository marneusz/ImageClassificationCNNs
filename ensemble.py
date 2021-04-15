import numpy as np
from keras.engine import training
from typing import List


def ensemble(models: List[training.Model], model_input: np.array, batch_size: int = 32):
    predictions_for_models = [model.predict(model_input, batch_size) for model in models]
    m, n = predictions_for_models[0].shape
    ensembled_predictions = np.zeros((m, n))

    for predictions in predictions_for_models:
        for i in range(m):
            for j in range(n):
                ensembled_predictions[i][j] += predictions[i][j]

    pred_classes_for_models = [np.argmax(preds, axis=1) for preds in predictions_for_models]
    return np.argmax(ensembled_predictions, axis=1), pred_classes_for_models

