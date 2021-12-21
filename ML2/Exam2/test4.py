import os
import cv2
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
import torch
from torch.utils.data import random_split, DataLoader
from torchsummary import summary
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
def get_data():

    x, y = [], []
    DATA_DIR = os.getcwd() + "/train/"
    RESIZE_TO = 400

    # count = 0
    for file in os.listdir(DATA_DIR):
        if file[-4:] == ".png":
            # count += 1
            # img = cv2.resize(cv2.imread(DATA_DIR + file), (RESIZE_TO, RESIZE_TO)).transpose((2, 0, 1))
            img = cv2.resize(cv2.imread(DATA_DIR + file), (RESIZE_TO, RESIZE_TO))
            # print(type(img))
            # img_n = img.astype('float') / 255.
            # input = torch.from_numpy(np.asarray(img_n))

            x.append(img)

            with open(DATA_DIR + file[:-4] + ".txt", "r") as s:
                label = []
                for line in s:
                    label.append(line.strip('\n'))
            y.append(label)

    # print(count)

    return x, y

x, y = get_data()
# x, y = np.array(x), np.array(y)
# print(type(x))


# Get data info
def get_info(y):
    rbc = 0
    d = 0
    g = 0
    t = 0
    r = 0
    s = 0
    l = 0
    for label in y:
        if "red blood cell" in label:  # 929
            rbc += 1
        if "difficult" in label:  # 255
            d += 1
        if "gametocyte" in label:  # 101
            g += 1
        if "trophozoite" in label:  # 462
            t += 1
        if "ring" in label:  # 204
            r += 1
        if "schizont" in label:  # 125
            s += 1
        if "leukocyte" in label:  # 59
            l += 1

    print("we have ", rbc, "red blood cell.")
    print("we have ", d, "difficult.")
    print("we have ", g, "gametocyte.")
    print("we have ", t, "trophozoite.")
    print("we have ", r, "ring.")
    print("we have ", s, "schizont.")
    print("we have ", l, "leukocyte.")

get_info(y)

# expand data set
def gen_data(x, y):

    h_generator = ImageDataGenerator(
        rotation_range=150,
        width_shift_range=0.2,
        # height_shift_range=0.2,
        # rescale=1. / 255,
        # shear_range=0.2,
        #zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    v_generator = ImageDataGenerator(
        rotation_range=150,
        height_shift_range=0.2,
        vertical_flip=True,
        fill_mode='nearest'
    )

    s_generator = ImageDataGenerator(
        rotation_range=150,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    z_generator = ImageDataGenerator(
        rotation_range=150,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # new_data_set = image_generator.flow(x, y, batch_size=1000)

    for index in range(len(y)):
        if "leukocyte" in y[index]:
            # 929//59 = 15
            for i in range(4):
                x.append(h_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(4):
                x.append(v_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(4):
                x.append(s_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(4):
                x.append(z_generator.random_transform(x[index]))
                y.append(y[index])
        elif "gametocyte" in y[index]:
            # 929//101 = 9
            for i in range(2):
                x.append(h_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(2):
                x.append(v_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(3):
                x.append(s_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(3):
                x.append(z_generator.random_transform(x[index]))
                y.append(y[index])
        elif "schizont" in y[index]:
            # 929//125 = 7
            for i in range(2):
                x.append(h_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(2):
                x.append(v_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(2):
                x.append(s_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(2):
                x.append(z_generator.random_transform(x[index]))
                y.append(y[index])
        elif "ring" in y[index]:
            # 929//204 = 4
            for i in range(1):
                x.append(h_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(1):
                x.append(v_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(1):
                x.append(s_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(1):
                x.append(z_generator.random_transform(x[index]))
                y.append(y[index])
        elif "difficult" in y[index]:
            # 929//255 = 3
            for i in range(1):
                x.append(h_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(1):
                x.append(v_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(1):
                x.append(s_generator.random_transform(x[index]))
                y.append(y[index])
            for i in range(1):
                x.append(z_generator.random_transform(x[index]))
                y.append(y[index])
    return x, y
    # return new_data_set

# x, y = gen_data(x, y)
# new_data_set = gen_data(x, y)
# x, y = next(new_data_set)
# print('After expand:')
# get_info(y)

# data preprocess
p = []
for img in x:
    # print(type(img))
    img = img.transpose((2, 0, 1))  # reference: https://www.programmersought.com/article/28392301362/  &&  https://www.programmersought.com/article/40575797138/
    img_n = img.astype('float') / 255.
    input = torch.from_numpy(np.array(img).copy())  # reference: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663
    p.append(input)

# one hot label
y = np.array(y)
one_hot = MultiLabelBinarizer(classes=["red blood cell", "difficult", "gametocyte", "trophozoite", "ring", "schizont", "leukocyte"])
y = one_hot.fit_transform(y)

# split data set and set up dataloader
data = list(zip(p, y))

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = random_split(data, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=1)
x_train, y_train = next(iter(train_loader))
# print(len(x_train))
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True, num_workers=1)
x_test, y_test = next(iter(test_loader))

# reference: https://stackoverflow.com/questions/57386851/how-to-get-entire-dataset-from-dataloader-in-pytorch

# put data to gpu
x_train, y_train = x_train.float().cuda(), y_train.float().cuda()
x_test, y_test = x_test.float().cuda(), y_test.float().cuda()


# set up
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#
LR = 5e-2
N_EPOCHS = 30
BATCH_SIZE = 32
DROPOUT = 0.05

# model
class CNN(nn.Module):
    def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, (3, 3))
            self.convnorm1 = nn.BatchNorm2d(16)
            self.pool1 = nn.MaxPool2d((2, 2))
            self.conv2 = nn.Conv2d(16, 32, (3, 3))
            self.convnorm2 = nn.BatchNorm2d(32)
            self.pool2 = nn.MaxPool2d((2, 2))
            self.conv3 = nn.Conv2d(32, 64, (3, 3))
            self.convnorm3 = nn.BatchNorm2d(64)
            self.pool3 = nn.MaxPool2d((2, 2))
            self.conv4 = nn.Conv2d(64, 64, (3,3))
            self.convnorm4 = nn.BatchNorm2d(64)
            self.pool4 = nn.MaxPool2d((2,2))
            self.linear1 = nn.Linear(64 * 23 * 23, 128)
            self.linear2 = nn.Linear(128, 7)
            # self.linear3 = nn.Linear(64, 7)
            self.drop = nn.Dropout(DROPOUT)
            self.sigmoid = nn.Sigmoid()
            self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.pool3(self.convnorm3(self.act(self.conv3(x))))
        x = self.pool4(self.convnorm4(self.act(self.conv4(x))))
        # x = self.linear3(self.drop(self.act(self.linear2(self.drop(self.act(self.linear1(x.view(len(x), -1))))))))
        x = self.linear2(self.drop(self.act(self.linear1(x.view(len(x), -1)))))
        return self.sigmoid(x)

model = CNN().to(device)
summary(model, (3, 400, 400)) # reference : https://stackoverflow.com/questions/55875279/how-to-get-an-output-dimension-for-each-layer-of-the-neural-network-in-pytorch
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()

# training
print("Starting training loop...")
for epoch in range(N_EPOCHS):
    loss_train = 0
    model.train()
    for batch in range(len(x_train) // BATCH_SIZE + 1):
        inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds])
        loss = criterion(logits, y_train[inds])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()

    with torch.no_grad():
        y_test_pred = model(x_test)
        loss = criterion(y_test_pred, y_test)
        loss_test = loss.item()

    # save model
    torch.save(model.state_dict(), "model_kaiqi.pt")
    model.eval()
    print("Epoch " + str(epoch) + " Loss: "+ str(loss_test))

