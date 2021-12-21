import os

os.system("sudo pip install cv2")

import numpy as np
import cv2
from keras.models import load_model


RESIZE_TO = 64

def predict(x):
    # Here x is a NumPy array. On the actual exam it will be a list of paths.
    # %% --------------------------------------------- Data Prep -------------------------------------------------------
    X = []

    for path in x:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img4 = cv2.resize(img, (RESIZE_TO, RESIZE_TO))
        X.append(img4)

    X = np.array(X)
    X = X.reshape(len(X), -1)
    X = X / 255
    # Write any data prep you used during training
    # %% --------------------------------------------- Predict ---------------------------------------------------------
    model = load_model('mlp_kaiqi.hdf5')

    y_pred = np.argmax(model.predict(X), axis=1)
    return y_pred, model