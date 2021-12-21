#%%
import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
import random
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.initializers import glorot_uniform
from sklearn.metrics import cohen_kappa_score, f1_score

#%%

# Load data
def get_data():

    x, y = [], []
    DATA_DIR = os.getcwd() + "/train/"
    RESIZE_TO = 64 # 64

    for file in os.listdir(DATA_DIR):
        if file[-4:] == ".png":
            img = cv2.imread(DATA_DIR + file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img4 = cv2.resize(img, (RESIZE_TO, RESIZE_TO))
            x.append(img4)

            with open(DATA_DIR + file[:-4] + ".txt", "r") as s:
                label = s.read()
            y.append(label)

    return x, y

x, y = get_data()

#%%
# Get data info
def get_info(y):
    rbc = 0
    r = 0
    s = 0
    t = 0
    for label in y:
        if label == "red blood cell":
            rbc += 1
        elif label == "ring":
            r += 1
        elif label == "schizont":
            s += 1
        else:
            t += 1
    
    print("we have ", rbc, "red blood cell.")
    print("we have ", r, "ring.")
    print("we have ", s, "schizont.")
    print("we have ", t, "trophozoite.")

get_info(y)

# 7000: 365: 133: 1109

#%%

# expand data set
def gen_data(x, y):

    image_generator = ImageDataGenerator(
        rotation_range=100,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    for index in range(len(y)):
        if y[index] == "ring":
            # 7000//365 = 19
            for i in range(19):
                x.append(image_generator.random_transform(x[index]))
                y.append(y[index])
        elif y[index] == "schizont":
            # 7000//133 = 52
            for i in range(52):
                x.append(image_generator.random_transform(x[index]))
                y.append(y[index])
        elif y[index] == "trophozoite":
            # 7000//1109 = 6
            for i in range(6):
                x.append(image_generator.random_transform(x[index]))
                y.append(y[index])
    
    return x, y

x, y = gen_data(x, y)

# reference: https://github.com/ictar/python-doc/blob/master/Machine%20Learning/%E4%BD%BF%E7%94%A8%E9%9D%9E%E5%B8%B8%E5%B0%91%E7%9A%84%E6%95%B0%E6%8D%AE%E6%9E%84%E5%BB%BA%E5%BC%BA%E5%A4%A7%E7%9A%84%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B.md


#%%

# save data
x, y = np.array(x), np.array(y)
le = LabelEncoder()
le.fit(["red blood cell", "ring", "schizont", "trophozoite"])
y = le.transform(y)

print(x.shape)
print(y.shape)

np.save("x_train.npy", x); np.save("y_train.npy", y)

#%%

# set up 
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
weight_init = glorot_uniform(seed=SEED)

# hyper parameter
LR = 1e-4
N_NEURONS = (1000, 500, 50) # (50 500 50)
N_EPOCHS = 1000 # 300
BATCH_SIZE = 64  # 64
# DROPOUT = 0.1

#%%

# split data
x, y = np.load("x_train.npy"), np.load("y_train.npy")

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=SEED, test_size=0.2, stratify=y)
x_train, x_test = x_train.reshape(len(x_train), -1), x_test.reshape(len(x_test), -1)
x_train, x_test = x_train/255, x_test/255
y_train, y_test = to_categorical(y_train, num_classes=4), to_categorical(y_test, num_classes=4)

# x_train, y_train = SMOTE().fit_sample(x_train, y_train)
# x_train, y_train = RandomOverSampler().fit_sample(x_train, y_train)
# x_train, y_train = ADASYN().fit_sample(x_train, y_train)
# x_train, y_train = BorderlineSMOTE().fit_sample(x_train, y_train)


print(x_train.shape)

#%%

# build and fit model
model = Sequential([
    Dense(N_NEURONS[0], activation="relu", kernel_initializer=weight_init),
    # Dropout(DROPOUT),
    BatchNormalization()
])

for n_neurons in N_NEURONS[1:]:
    model.add(Dense(n_neurons, activation="relu", kernel_initializer=weight_init))
    # model.add(Dropout(DROPOUT))
    model.add(BatchNormalization())

model.add(Dense(4, activation="softmax", kernel_initializer=weight_init))

model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

my_callbacks = [
    EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=100),
    ModelCheckpoint(filepath="mlp_kaiqi.hdf5", monitor="val_loss", save_best_only=True)
]

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=N_EPOCHS,
          validation_data=(x_test, y_test),
          callbacks=my_callbacks)

print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1), average='macro'))
# %%
