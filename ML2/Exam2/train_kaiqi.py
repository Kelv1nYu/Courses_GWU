import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score
from torchvision import transforms, utils

class myDataSet(Dataset):
    def __init__(self, DATA_DIR, transform=None):
        self.img_path = []
        self.label_path = []
        self.transform = transform

        for file in os.listdir(DATA_DIR):
            if file[-4:] == '.png':
                self.img_path.append(DATA_DIR + file)
                self.label_path.append(DATA_DIR + file[:-4] + '.txt')

    def __getitem__(self, index):
        # print(self.img_path[index])
        RESIZE_TO = 512
        img = cv2.imread(self.img_path[index])
        img = cv2.resize(img, (RESIZE_TO, RESIZE_TO)).transpose((2, 0, 1))
        img = img.astype('float') / 255.

        img = torch.from_numpy(np.array(img))
        label = torch.from_numpy(np.array(self.one_hot(self.label_path[index])))

        return img, label

    def __len__(self):
        return len(self.img_path)

    def one_hot(self, label_path):
        label = [0.0] * 7
        with open(label_path, 'r') as f:
            for line in f:
                type = line.strip('\n')
                if type == 'red blood cell':
                    label[0] = 1
                elif type == 'difficult':
                    label[1] = 1
                elif type == 'gametocyte':
                    label[2] = 1
                elif type == 'trophozoite':
                    label[3] = 1
                elif type == 'ring':
                    label[4] = 1
                elif type == 'schizont':
                    label[5] = 1
                else:
                    label[6] = 1

        return label


DATA_DIR = os.getcwd() + '/train/'
data = myDataSet(DATA_DIR)

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = random_split(data, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=1)
# x_train, y_train = next(iter(train_loader))

test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True, num_workers=1)
# x_test, y_test = next(iter(test_loader))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

LR = 5e-2
N_EPOCHS = 18
BATCH_SIZE = 33
DROPOUT = 0.05


def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100 * accuracy_score(y.cpu().numpy(), pred_labels)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, (5, 5), stride=5)  # output (n_examples, 16, 240, 320)
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))  # output (n_examples, 16, 120, 160)
        self.conv2 = nn.Conv2d(16, 32, (5, 5), stride=5)  # output (n_examples, 32, 24, 32)
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2, 2))  # output (n_examples, 32, 12, 16)
        self.linear1 = nn.Linear(32 * 12 * 16, 400)  # input will be flattened to (n_examples, 32 * 12 * 16)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(DROPOUT)
        self.linear2 = nn.Linear(400, 7)
        self.act1 = torch.relu
        self.act2 = torch.sigmoid

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act1(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act1(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act1(self.linear1(x.view(len(x), -1)))))
        return self.act2(self.linear2(x))

        # return self.sigmoid(x)

model = CNN().type('torch.DoubleTensor').to(device)
# model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()
#criterion = nn.BCELoss()

print("Starting training loop...")

for epoch in range(N_EPOCHS):

    loss_train = 0
    model.train()
    # for images, labels in train_loader:
    # for batch in range(len(x_train) // BATCH_SIZE + 1):
        # inds = slice(batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE)
        # if torch.cuda.is_available():
        #
        #     images, labels = images.to(device), labels.to(device)
        #     print(images.shape)
        # optimizer.zero_grad()
        # logits = model(images)
        # loss = criterion(logits, labels)
        # loss.backward()
        # optimizer.step()
        # loss_train += loss.item()

    for i, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.cuda(), y_train.cuda()
        y_train = y_train.float()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()  # updates weights
        loss_train += loss.item()

    print("Epoch {} | Train Loss {:.5f}".format(
        epoch, loss_train))

print('Testing loop')

loss_test = 0
# model.load_state_dict(torch.load("model_kghimire92.pt"))
model.eval()  # Deactivates Dropout and makes BatchNorm use mean and std estimates computed during training
with torch.no_grad():  # The code inside will run without Autograd, which reduces memory usage, speeds up
    # computations and makes sure the model can't use the test data to learn
    for i, (x_test, y_test) in enumerate(test_loader):
        x_test, y_test = x_test.cuda(), y_test.cuda()
        y_test = y_test.float()
        y_test_pred = model(x_test)
        loss = criterion(y_test_pred, y_test)
        loss_test += loss.item()
print("Test Loss {:.5f}".format(loss_test))

    # model.eval()
    # correct = 0
    # with torch.no_grad():
    #     for images, labels in test_loader:
    #         if torch.cuda.is_available():
    #             images, labels = images.to(device), labels.to(device)
    #     y_test_pred = model(images)
    #     # print(y_test_pred)
    #     loss = criterion(y_test_pred, labels)
    #     loss_test = loss.item()
    #     # torch.save(model.state_dict(), "model_whu369.pt")
    # print('Test set: Average loss: (:.3f), Accuracy: {}/{} ({:.0f}%)\n'.format(loss_test, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

        # print("Perfect Result: Loss ---> "+str(loss_test))

# reference : https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch






