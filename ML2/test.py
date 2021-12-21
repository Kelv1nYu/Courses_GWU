# %% -------------------------------------------------------------------------------------------------------------------
# --------------------------------------- Imports -------------------------------------------------------------------
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data.sampler import SubsetRandomSampler

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# %% -------------------------------------- Data Prep ------------------------------------------------------------------

if "train" not in os.listdir():
    os.system("wget https://storage.googleapis.com/exam-deep-learning/train-Exam2.zip")
    os.system("unzip train-Exam2.zip")

DATA_DIR = os.getcwd() + "/train/"

RESIZE_TO = 400

x, y = [], []
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".png"]:
    # print(DATA_DIR + path)

    img = cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_TO, RESIZE_TO))
    img = img.transpose((2, 0, 1))
    img = img.astype('float') / 255.0
    x.append(img)

    # x.append(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_TO, RESIZE_TO)).transpose((2, 0, 1)))
    with open(DATA_DIR + path[:-4] + ".txt", "r") as s:
        # label = s.read ()
        s1 = s.readlines()
        label = []
        for sub in s1:
            # print (sub.strip())
            # print(sub)
            label.append(sub.strip().split(","))
        # print(np.array(label).reshape(1,-1)[0])
    y.append(np.array(label).reshape(1, -1)[0])

x, y = np.array(x), np.array(y)
# create multilabel binarizer object #
one_hot = MultiLabelBinarizer(
    classes=["red blood cell", "difficult", "gametocyte", "trophozoite", "ring", "schizont", "leukocyte"])
y = one_hot.fit_transform(y).astype(np.double)
dataset = list(zip(x, y))

############ split data into training and test #######
# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# test_size = 0.2
# split = int(np.floor(test_size * dataset_size))
#
# np.random.seed(402)
# np.random.shuffle(indices)
# train_indices, test_indices = indices[split:], indices[:split]
# train_sampler = SubsetRandomSampler(train_indices)
# test_sampler = SubsetRandomSampler(test_indices)
#
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
#
# test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=1)
x_train, y_train = next(iter(train_loader))

test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True, num_workers=1)
x_test, y_test = next(iter(test_loader))

LR = 0.001
N_EPOCHS = 18
BATCH_SIZE = 33
DROPOUT = 0.05


    # %% ----------------------------------- Helper Functions --------------------------------------------------------------
def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100 * accuracy_score(y.cpu().numpy(), pred_labels)

    # %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 18, (3, 3))  # output (n_examples, 18, 396, 396) Output dim = (400-3)/1 + 1 = 398
        self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 18, 199, 199) Output dim = (398-2)/2 + 1 = 199
        self.conv2 = nn.Conv2d(18, 36, (3, 3))  # output (n_examples, 36, 197, 197) Output dim = (199-3)/1 + 1 = 197
        self.pool2 = nn.MaxPool2d((2, 2), ceil_mode=False)  # output (n_examples, 36, 98, 98) Output dim = (197 - 2)/2 + 1 = 98
        self.conv3 = nn.Conv2d(36, 72, (3, 3)) # output (n_examples, 72, 96, 96) Output dim = (98 - 3)/1 + 1 = 96
        self.pool3 = nn.MaxPool2d((2, 2)) # output (n_examples, 72, 48, 48) Output dim = (96 - 2)/2 + 1 = 48
        self.conv4 = nn.Conv2d(72, 72, (3,3)) # output (n_examples, 72, 46, 46) Output dim = (48 - 3)/1 + 1 = 46
        self.pool4 = nn.MaxPool2d((2,2)) # output (n_example, 72, 23, 23) Output dim = (46 - 2)/2 + 1 = 23
        self.linear1 = nn.Linear(72 * 23 * 23, 128)  # input will be flattened to (n_examples, 72 * 23 * 23)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 7)
        self.drop1 = nn.Dropout(DROPOUT)
        self.drop2 = nn.Dropout(DROPOUT)
        self.sigmoid = nn.Sigmoid()
        self.act = torch.relu

    def forward(self, x):
        x = self.drop1(self.pool1(self.act(self.conv1(x))))
        x = self.drop1(self.pool2(self.act(self.conv2(x))))
        x = self.drop1(self.pool3(self.act(self.conv3(x))))
        x = self.drop1(self.pool4(self.act(self.conv4(x))))
        x = self.linear3(self.drop2(self.act(self.linear2(self.drop2(self.act(self.linear1(x.view(len(x), -1))))))))
        return self.sigmoid(x)

    # %% -------------------------------------- Training Prep ----------------------------------------------------------
    # format the type from float to double
model = CNN().type('torch.DoubleTensor').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCELoss()

    # %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
for epoch in range(N_EPOCHS):

    loss_train = 0
    model.train()
    for batch in range(len(x_train) // BATCH_SIZE):
        inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
        optimizer.zero_grad()
        logits = model(x_train[inds].cuda())
        loss = criterion(logits, y_train[inds].cuda())
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
    with torch.no_grad():
        y_test_pred = model(x_test.cuda())
        print(y_test_pred)
        loss = criterion(y_test_pred, y_test.cuda())
        loss_test = loss.item()
        # torch.save(model.state_dict(), "model_whu369.pt")
    model.eval()
    print("Perfect Result: Loss ---> "+str(loss_test))